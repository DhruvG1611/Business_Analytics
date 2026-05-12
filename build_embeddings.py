"""
build_embeddings.py
────────────────────────────────────────────────────────────────────────────────
Offline script — run once (or whenever csm_enterprise.yaml changes).

What it builds
──────────────
Two FAISS indexes are persisted to ./embeddings/:

  1. metrics_index   — one vector per CSM metric + dimension
  2. patterns_index  — one vector per CSM intent_pattern

Each index is accompanied by a JSON metadata sidecar so the retriever can
reconstruct rich result objects without touching the YAML files at query time.

Embedding model
───────────────
Uses sentence-transformers "all-MiniLM-L6-v2" (384-dim, fast, fully offline).

Usage
─────
    python build_embeddings.py                        # uses defaults
    python build_embeddings.py --csm csm_enterprise.yaml --out ./embeddings
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with:  pip install faiss-cpu sentence-transformers tqdm"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text serialisers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_to_text(metric_id: str, node: dict) -> str:
    """Produces a rich natural-language string for a CSM metric."""
    parts = [
        f"metric: {metric_id}",
        f"label: {node.get('label', '')}",
    ]
    if node.get('description'):
        parts.append(f"description: {node['description']}")
    if node.get('compute'):
        parts.append(f"compute: {node['compute']}")
    sources = node.get('sources', [])
    if sources:
        parts.append(f"sources: {', '.join(sources)}")
    return "\n".join(parts)


def _dimension_to_text(dim_id: str, node: dict) -> str:
    """Encodes a CSM dimension into a searchable string."""
    parts = [
        f"dimension: {dim_id}",
        f"label: {node.get('label', '')}",
        f"source: {node.get('source', '')}.{node.get('column', '')}",
        f"type: {node.get('type', '')}",
    ]
    if node.get('sample_values'):
        parts.append("samples: " + ", ".join(str(v) for v in node['sample_values']))
    return "\n".join(parts)


def _pattern_to_text(pattern: dict) -> str:
    """
    Encodes a CSM intent_pattern as a flat string for embedding.
    Includes partition_by so windowed queries score correctly.
    """
    parts = [f"pattern: {pattern.get('pattern', '')}"]

    if pattern.get('metric'):
        parts.append(f"metric: {pattern['metric']}")

    dims = pattern.get('dimensions', [])
    if dims:
        parts.append(f"dimensions: {', '.join(dims)}")

    # Include partition_by from the window block in the searchable text
    window = pattern.get('window', {})
    partition_by = window.get('partition_by', '')
    if partition_by and partition_by != '_none_':
        parts.append(f"partition_by: {partition_by}")

    if pattern.get('sort'):
        parts.append(f"sort: {pattern['sort']}")
    if pattern.get('limit'):
        parts.append(f"limit: {pattern['limit']}")
    if pattern.get('mode'):
        parts.append(f"mode: {pattern['mode']}")

    filters = pattern.get('filters', [])
    if filters:
        fstr = "; ".join(
            f"{f.get('field')}={f.get('values', [])}" for f in filters
        )
        parts.append(f"filters: {fstr}")

    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Index builders
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics_index(
    csm: dict,
    model: SentenceTransformer,
    out_dir: Path,
) -> None:
    """
    Encodes every CSM metric and dimension into a single IndexFlatIP.
    Writes:
        embeddings/metrics_index.faiss
        embeddings/metrics_meta.json
    """
    texts = []
    meta  = []

    # Metrics
    for m_id, m_node in csm.get('metrics', {}).items():
        if m_node is None:
            continue
        texts.append(_metric_to_text(m_id, m_node))
        meta.append({
            "type":        "metric",
            "key":         m_id,
            "label":       m_node.get('label', ''),
            "compute":     m_node.get('compute', ''),
            "sources":     m_node.get('sources', []),
            "join_path":   m_node.get('join_path'),
            "description": m_node.get('description', ''),
        })

    # Dimensions
    for d_id, d_node in csm.get('dimensions', {}).items():
        if d_node is None:
            continue
        texts.append(_dimension_to_text(d_id, d_node))
        meta.append({
            "type":     "dimension",
            "key":      d_id,
            "label":    d_node.get('label', ''),
            "source":   d_node.get('source', ''),
            "column":   d_node.get('column', ''),
            "col_type": d_node.get('type', ''),
        })

    print(f"  Encoding {len(texts)} metric/dimension entries …")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "metrics_index.faiss"))
    with open(out_dir / "metrics_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ metrics_index: {index.ntotal} vectors  →  {out_dir}/metrics_index.faiss")


def build_patterns_index(
    csm: dict,
    model: SentenceTransformer,
    out_dir: Path,
) -> None:
    """
    Encodes every CSM intent_pattern.
    Writes:
        embeddings/patterns_index.faiss
        embeddings/patterns_meta.json
    """
    patterns = csm.get('intent_patterns', [])
    if not patterns:
        print("  ⚠  No intent_patterns found in CSM — skipping patterns index.")
        return

    texts = [_pattern_to_text(p) for p in patterns]

    # Save the full window block so rag_retriever can extract partition_by
    meta = [
        {
            "pattern":          p.get('pattern', ''),
            "metric":           p.get('metric'),
            "dimensions":       p.get('dimensions', []),
            "filters":          p.get('filters', []),
            "sort":             p.get('sort'),
            "limit":            p.get('limit'),
            "mode":             p.get('mode'),
            "having_threshold": p.get('having_threshold'),
            "window":           p.get('window', {}),   # ← partition_by lives here
        }
        for p in patterns
    ]

    print(f"  Encoding {len(texts)} intent patterns …")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "patterns_index.faiss"))
    with open(out_dir / "patterns_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ patterns_index: {index.ntotal} vectors  →  {out_dir}/patterns_index.faiss")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build FAISS embeddings for CSM")
    parser.add_argument("--csm", default="csm_enterprise.yaml", help="Path to CSM YAML")
    parser.add_argument("--out", default="./embeddings",        help="Output directory")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSM (utf-8 so box-drawing chars in comments don't crash on Windows)
    print(f"\n[1/4] Loading CSM YAML …")
    with open(args.csm, encoding="utf-8") as f:
        csm = yaml.safe_load(f)
    print(f"      metrics:         {len(csm.get('metrics', {}))}")
    print(f"      dimensions:      {len(csm.get('dimensions', {}))}")
    print(f"      intent_patterns: {len(csm.get('intent_patterns', []))}")

    # Load embedding model
    print(f"\n[2/4] Loading embedding model '{args.model}' …")
    model = SentenceTransformer(args.model)
    print(f"      Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Build indexes
    print(f"\n[3/4] Building metrics + dimensions index …")
    build_metrics_index(csm, model, out_dir)

    print(f"\n[4/4] Building intent patterns index …")
    build_patterns_index(csm, model, out_dir)

    # Manifest
    manifest = {
        "model":           args.model,
        "csm_file":        args.csm,
        "metrics_entries": len(csm.get('metrics', {})) + len(csm.get('dimensions', {})),
        "pattern_entries": len(csm.get('intent_patterns', [])),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Done — all indexes written to '{out_dir}/'")
    print("   Re-run whenever csm_enterprise.yaml changes.")


if __name__ == "__main__":
    main()