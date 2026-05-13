"""
retriever_context.py
--------------------
Single retrieve() call per question. The result object is computed once
and passed through pipeline state to both the LLM context builder and
the hint extractor, eliminating the double-retrieve bug.
"""
from core.rag_retriever import retrieve, build_retrieved_context

# Threshold for extracting partition_by from a matched pattern.
# Set to 0.72 to match _PATTERN_THRESH in rag_retriever.py — MiniLM
# cosine scores top out ~0.75 on paraphrases so 0.82 was unreachable.
_PARTITION_BY_THRESH = 0.65


def retrieve_once(question: str):
    """Call once per question. Pass the result to both helpers below."""
    return retrieve(question)


def build_rag_context(retrieve_result, glossary: dict) -> str:
    """
    Builds a compact LLM prompt context from an already-fetched retrieve result.
    Sends only relevant glossary entries, never the full BGO dump.
    """
    if retrieve_result.retrieved and (
        retrieve_result.top_metrics or retrieve_result.top_patterns
    ):
        retrieved_ctx = build_retrieved_context(retrieve_result)

        relevant_keys = set()
        for m in (retrieve_result.top_metrics or []):
            relevant_keys.add(getattr(m, 'key', None))
        for p in (retrieve_result.top_patterns or []):
            relevant_keys.add(getattr(p, 'metric', None))
        relevant_keys.discard(None)

        metrics_excerpt = {
            k: {"label": v.get("label", k), "calculation": v.get("calculation", "")}
            for k, v in glossary.get("metrics", {}).items()
            if k in relevant_keys
        }
        excerpt = f"\n\n## RELEVANT METRICS\n{metrics_excerpt}" if metrics_excerpt else ""
        return retrieved_ctx + excerpt

    compact = {k: v.get("label", k) for k, v in glossary.get("metrics", {}).items()}
    return f"## AVAILABLE METRICS\n{compact}"


def extract_rag_hints(retrieve_result) -> dict:
    """
    Extracts high-confidence hints from an already-fetched retrieve result.
    Passes partition_by and pattern_score through so intent_parser and
    graph_resolver can use the correct PARTITION BY column.
    """
    if not retrieve_result.retrieved:
        return {
            "best_metric_key": None,
            "pattern_intent":  None,
            "pattern_score":   0.0,
            "retrieved":       False,
        }

    pattern_intent = None
    pattern_score  = 0.0

    if retrieve_result.best_pattern:
        p             = retrieve_result.best_pattern
        pattern_score = p.score

        # Fire at _PARTITION_BY_THRESH - lowered from 0.82 so MiniLM can reach it
        if p.score >= _PARTITION_BY_THRESH:
            # Weighted scoring: favor metric matches if the pattern suggests a different metric
            m_score = retrieve_result.best_metric.score if retrieve_result.best_metric else 0.0
            best_m_key = retrieve_result.best_metric.key if retrieve_result.best_metric else ""
            
            if p.metric == best_m_key or p.score > (m_score * 1.2):
                # Extract partition_by from the window block stored on the PatternHit
                window       = getattr(p, 'window', {}) or {}
                raw_partition = window.get('partition_by', '_none_')
                partition_by  = [] if raw_partition in ('_none_', None, '') \
                                else [raw_partition]

                pattern_intent = {
                    "metric":           p.metric,
                    "dimensions":       p.dimensions,
                    "filters":          p.filters or [],
                    "sort":             p.sort,
                    "limit":            p.limit,
                    "mode":             p.mode,
                    "having_threshold": getattr(p, "having_threshold", None),
                    "partition_by":     partition_by,   # <- the critical field
                }
            else:
                print(f"  [RAG] Pattern suppressed: metric score {m_score:.3f} weighted > pattern score {p.score:.3f}")

    return {
        "best_metric_key": retrieve_result.best_metric.key
                           if retrieve_result.best_metric else None,
        "pattern_intent":  pattern_intent,
        "pattern_score":   pattern_score,
        "retrieved":       True,
        "metric_confidence": getattr(retrieve_result, "metric_confidence", "none"),
    }