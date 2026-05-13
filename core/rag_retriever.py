"""
rag_retriever.py
────────────────────────────────────────────────────────────────────────────────
Runtime retrieval module.  Loaded once at startup; the FAISS indexes and
metadata are kept in memory for the lifetime of the process.

Public API (consumed by retriever_context.py)
─────────────────────────────────────────────
    result = retrieve(question: str) -> RetrievalResult
    text   = build_retrieved_context(result: RetrievalResult) -> str

RetrievalResult fields
──────────────────────
    retrieved     bool          — False if indexes are unavailable / query failed
    best_metric   MetricHit?    — highest-scoring metric entry
    top_metrics   list[MetricHit]
    best_pattern  PatternHit?   — highest-scoring intent pattern (score ≥ threshold)
    top_patterns  list[PatternHit]

MetricHit  : key, label, score, compute, sources, join_path, description
PatternHit : pattern, metric, dimensions, filters, sort, limit, mode,
             having_threshold, score

Configuration
─────────────
Set environment variables to override defaults:

    RAG_EMBEDDINGS_DIR   path to the embeddings/ folder   (default: ./embeddings)
    RAG_MODEL_OVERRIDE   sentence-transformers model name  (uses manifest.json by default)
    RAG_TOP_K            how many candidates to retrieve   (default: 5)
    RAG_PATTERN_THRESH   min score for a pattern to count  (default: 0.82)
    RAG_METRIC_THRESH    min score for a metric to count   (default: 0.40)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── lazy heavy imports ────────────────────────────────────────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with:  pip install faiss-cpu sentence-transformers"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (overridable via env vars)
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDINGS_DIR   = Path(os.getenv("RAG_EMBEDDINGS_DIR",  "./embeddings"))
_MODEL_OVERRIDE   = os.getenv("RAG_MODEL_OVERRIDE",       "")          # empty = use manifest
_TOP_K            = int(os.getenv("RAG_TOP_K",            "5"))
_PATTERN_THRESH   = float(os.getenv("RAG_PATTERN_THRESH", "0.65"))
_METRIC_THRESH    = float(os.getenv("RAG_METRIC_THRESH",  "0.65"))


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricHit:
    key:         str
    label:       str
    score:       float
    compute:     str        = ""
    sources:     list       = field(default_factory=list)
    join_path:   Optional[list] = None
    description: str        = ""
    entry_type:  str        = "metric"   # "metric" | "dimension"


@dataclass
class PatternHit:
    pattern:          str
    metric:           Optional[str]
    dimensions:       list  = field(default_factory=list)
    filters:          list  = field(default_factory=list)
    sort:             Optional[str]  = None
    limit:            Optional[int]  = None
    mode:             Optional[str]  = None
    having_threshold: Optional[dict] = None
    score:            float = 0.0


@dataclass
class RetrievalResult:
    retrieved:    bool
    best_metric:  Optional[MetricHit]  = None
    top_metrics:  list[MetricHit]      = field(default_factory=list)
    best_pattern: Optional[PatternHit] = None
    top_patterns: list[PatternHit]     = field(default_factory=list)
    metric_confidence: str = "none"


# ─────────────────────────────────────────────────────────────────────────────
# Index store  (singleton, loaded lazily on first retrieve() call)
# ─────────────────────────────────────────────────────────────────────────────

class _IndexStore:
    """Holds both FAISS indexes and their metadata in memory."""

    _instance: Optional["_IndexStore"] = None

    def __init__(self):
        self._loaded       = False
        self._model        = None
        self._metrics_idx  = None
        self._metrics_meta = []
        self._patterns_idx = None
        self._patterns_meta= []

    # ── singleton accessor ────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "_IndexStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── loader ────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Loads indexes from disk.  Returns True on success, False if the
        embeddings directory or required files don't exist yet (caller should
        fall back gracefully instead of crashing).
        """
        if self._loaded:
            return True

        d = _EMBEDDINGS_DIR
        manifest_path  = d / "manifest.json"
        metrics_faiss  = d / "metrics_index.faiss"
        metrics_meta   = d / "metrics_meta.json"
        patterns_faiss = d / "patterns_index.faiss"
        patterns_meta  = d / "patterns_meta.json"

        missing = [p for p in [metrics_faiss, metrics_meta, patterns_faiss, patterns_meta] if not p.exists()]
        if missing:
            print(
                f"[rag_retriever] ⚠  Embeddings not found in '{d}'.\n"
                f"  Missing: {[str(p) for p in missing]}\n"
                "  Run  python build_embeddings.py  to generate them."
            )
            return False

        # ── determine which embedding model to load ───────────────────────────
        model_name = _MODEL_OVERRIDE
        if not model_name and manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            model_name = manifest.get("model", "all-MiniLM-L6-v2")
        model_name = model_name or "all-MiniLM-L6-v2"

        print(f"[rag_retriever] Loading embedding model '{model_name}' …")
        self._model = SentenceTransformer(model_name)

        # ── load FAISS indexes ────────────────────────────────────────────────
        self._metrics_idx  = faiss.read_index(str(metrics_faiss))
        self._patterns_idx = faiss.read_index(str(patterns_faiss))

        with open(metrics_meta)  as f: self._metrics_meta  = json.load(f)
        with open(patterns_meta) as f: self._patterns_meta = json.load(f)

        self._loaded = True
        print(
            f"[rag_retriever] [OK] Loaded - "
            f"{self._metrics_idx.ntotal} metric/dim vectors, "
            f"{self._patterns_idx.ntotal} pattern vectors."
        )
        return True

    # ── query helpers ─────────────────────────────────────────────────────────
    def _encode(self, text: str) -> np.ndarray:
        """Encodes a single query string; returns shape (1, dim) float32."""
        vec = self._model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def search_metrics(self, question: str, k: int = _TOP_K) -> list[MetricHit]:
        """Returns top-k metric/dimension hits for the question."""
        if not self._loaded:
            return []
        vec = self._encode(question)
        scores, indices = self._metrics_idx.search(vec, k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self._metrics_meta[idx]
            hits.append(MetricHit(
                key         = m["key"],
                label       = m.get("label", ""),
                score       = float(score),
                compute     = m.get("compute", ""),
                sources     = m.get("sources", []),
                join_path   = m.get("join_path"),
                description = m.get("description", ""),
                entry_type  = m.get("type", "metric"),
            ))
        return hits

    def search_patterns(self, question: str, k: int = _TOP_K) -> list[PatternHit]:
        """Returns top-k pattern hits for the question."""
        if not self._loaded:
            return []
        vec = self._encode(question)
        scores, indices = self._patterns_idx.search(vec, k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            p = self._patterns_meta[idx]
            hits.append(PatternHit(
                pattern          = p.get("pattern", ""),
                metric           = p.get("metric"),
                dimensions       = p.get("dimensions", []),
                filters          = p.get("filters", []),
                sort             = p.get("sort"),
                limit            = p.get("limit"),
                mode             = p.get("mode"),
                having_threshold = p.get("having_threshold"),
                score            = float(score),
            ))
        return hits


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(question: str) -> RetrievalResult:
    """
    Main entry point.  Searches both indexes and returns a RetrievalResult.

    The caller (retriever_context.py) can then:
      • Use result.best_metric.key  as a strong hint for the LLM prompt
      • Use result.best_pattern     as a near-complete intent override (if score ≥ 0.82)
      • Use result.top_metrics      to build a richer schema context string
    """
    store = _IndexStore.get()
    if not store.load():
        # Embeddings not built yet — degrade gracefully
        return RetrievalResult(retrieved=False)

    # ── metric / dimension search ─────────────────────────────────────────────
    all_hits   = store.search_metrics(question, k=_TOP_K)
    # Separate metrics from dimensions; keep only metric hits for best_metric
    metric_hits = [h for h in all_hits if h.entry_type == "metric" and h.score >= _METRIC_THRESH]
    top_metrics = sorted(metric_hits, key=lambda h: h.score, reverse=True)
    best_metric = top_metrics[0] if top_metrics else None

    # ── pattern search ────────────────────────────────────────────────────────
    pattern_hits = store.search_patterns(question, k=_TOP_K)
    top_patterns = sorted(pattern_hits, key=lambda h: h.score, reverse=True)
    # DEBUG — remove after fix is confirmed
    for p in top_patterns[:3]:
        print(f"  [RAG-DEBUG] pattern score={p.score:.3f}  pattern={p.pattern!r}")
    # best_pattern is only surfaced if it clears the confidence threshold
    best_pattern = top_patterns[0] if (top_patterns and top_patterns[0].score >= _PATTERN_THRESH) else None

    retrieved = bool(best_metric or best_pattern)

    if best_metric:
        print(f"  [RAG] best_metric  -> {best_metric.key!r}  (score={best_metric.score:.3f})")
    if best_pattern:
        print(f"  [RAG] best_pattern -> {best_pattern.pattern!r}  (score={best_pattern.score:.3f})")

    confidence = "none"
    if best_metric:
        if best_metric.score >= 0.75:
            confidence = "high"
        else:
            confidence = "low"

    return RetrievalResult(
        retrieved    = retrieved,
        best_metric  = best_metric,
        top_metrics  = top_metrics,
        best_pattern = best_pattern,
        top_patterns = top_patterns,
        metric_confidence = confidence,
    )


def build_retrieved_context(result: RetrievalResult) -> str:
    """
    Serialises a RetrievalResult into a human-readable string suitable for
    injection into the LLM's system/context prompt.

    Used by retriever_context.build_rag_context().
    """
    if not result.retrieved:
        return ""

    lines = ["## RETRIEVED CONTEXT (from semantic search)\n"]

    # ── top metrics ───────────────────────────────────────────────────────────
    if result.top_metrics:
        lines.append("### Likely relevant metrics")
        for h in result.top_metrics[:3]:    # cap at 3 to keep the prompt lean
            lines.append(
                f"- **{h.key}** (score={h.score:.2f})\n"
                f"  label: {h.label}\n"
                f"  compute: `{h.compute}`\n"
                f"  sources: {', '.join(h.sources)}"
                + (f"\n  -> {h.description}" if h.description else "")
            )
        lines.append("")

    # ── best pattern (high-confidence template) ───────────────────────────────
    if result.best_pattern:
        p = result.best_pattern
        lines.append("### Closest matching intent pattern")
        lines.append(f"  pattern : \"{p.pattern}\"")
        lines.append(f"  metric  : {p.metric}")
        if p.dimensions:
            lines.append(f"  dims    : {', '.join(p.dimensions)}")
        if p.sort:
            lines.append(f"  sort    : {p.sort}  limit: {p.limit}")
        if p.filters:
            lines.append(f"  filters : {p.filters}")
        if p.mode:
            lines.append(f"  mode    : {p.mode}")
        lines.append(f"  (similarity score: {p.score:.3f})")
        lines.append("")

    # ── runner-up patterns (lower confidence — informational only) ────────────
    runners = [p for p in result.top_patterns[1:3] if p.score >= 0.60]
    if runners:
        lines.append("### Other candidate patterns")
        for p in runners:
            lines.append(f"  - [{p.score:.2f}] {p.pattern!r}  ->  metric={p.metric}")
        lines.append("")

    return "\n".join(lines)