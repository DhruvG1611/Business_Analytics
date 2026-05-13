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

    print(f"  [OK] metrics_index: {index.ntotal} vectors  ->  {out_dir}/metrics_index.faiss")


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
    patterns = csm.get('all_patterns') or csm.get('intent_patterns', [])
    if not patterns:
        print("  ⚠  No intent_patterns found in CSM/BGO — skipping patterns index.")
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

    print(f"  [OK] patterns_index: {index.ntotal} vectors  ->  {out_dir}/patterns_index.faiss")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build(csm_path="csm_enterprise.yaml", out_path="./embeddings", model_name="all-MiniLM-L6-v2"):
    import os
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSM (utf-8 so box-drawing chars in comments don't crash on Windows)
    print(f"\n[1/4] Loading CSM YAML …")
    with open(csm_path, encoding="utf-8") as f:
        csm = yaml.safe_load(f)
    print(f"      metrics:         {len(csm.get('metrics', {}))}")
    print(f"      dimensions:      {len(csm.get('dimensions', {}))}")
    print(f"      intent_patterns: {len(csm.get('intent_patterns', []))}")

    # Load embedding model
    print(f"\n[2/4] Loading embedding model '{model_name}' …")
    model = SentenceTransformer(model_name)
    print(f"      Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Build indexes
    print(f"\n[3/4] Building metrics + dimensions index …")
    build_metrics_index(csm, model, out_dir)

    # Load BGO patterns if they exist
    print(f"\n[4/4] Building intent patterns index (CSM + BGO) …")
    all_patterns = csm.get('intent_patterns', [])
    try:
        with open("bgo.yaml", encoding="utf-8") as f:
            bgo = yaml.safe_load(f)
            bgo_patterns = bgo.get('intent_patterns', [])
            all_patterns.extend(bgo_patterns)
    except Exception as e:
        print(f"      Note: Could not load extra patterns from bgo.yaml: {e}")

    # Use a combined dict or just pass the list if we modify build_patterns_index
    # For minimal impact, let's just temporarily put them in the csm dict
    csm['all_patterns'] = all_patterns
    build_patterns_index(csm, model, out_dir)

    # Manifest
    manifest = {
        "model":           model_name,
        "csm_file":        csm_path,
        "metrics_entries": len(csm.get('metrics', {})) + len(csm.get('dimensions', {})),
        "pattern_entries": len(csm.get('intent_patterns', [])),
        "built_at":        os.path.getmtime(csm_path) if os.path.exists(csm_path) else 0,
        "bgo_built_at":    os.path.getmtime("bgo.yaml") if os.path.exists("bgo.yaml") else 0
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] all indexes written to '{out_dir}/'")
    print("   Re-run whenever csm_enterprise.yaml changes.")

def embeddings_are_stale(csm_path="csm_enterprise.yaml", out_path="./embeddings") -> bool:
    import os
    manifest_path = Path(out_path) / "manifest.json"
    if not manifest_path.exists():
        return True
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        csm_mtime = os.path.getmtime(csm_path) if os.path.exists(csm_path) else 0
        bgo_mtime = os.path.getmtime("bgo.yaml") if os.path.exists("bgo.yaml") else 0
        return manifest.get("built_at", 0) != csm_mtime or manifest.get("bgo_built_at", 0) != bgo_mtime
    except Exception:
        return True

def build_embeddings_if_stale(force=False, csm_path="csm_enterprise.yaml", out_path="./embeddings"):
    if force or embeddings_are_stale(csm_path, out_path):
        print("\n  [Embeddings] CSM/BGO changed or FAISS missing. Rebuilding...")
        build(csm_path=csm_path, out_path=out_path)
        
        # Reset the RAG singleton to force reload in runtime
        try:
            from core.rag_retriever import _IndexStore
            _IndexStore._instance = None
            print("  [Embeddings] RAG singleton reset.")
        except ImportError:
            pass

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
    build(csm_path=args.csm, out_path=args.out, model_name=args.model)

if __name__ == "__main__":
    main()