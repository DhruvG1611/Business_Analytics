"""
pipeline.py
-----------
Clean LangChain pipeline with:
  - Single retrieve() call threaded through all steps
  - Pre-built static schema context (zero cost per query)
  - Error recovery at each step
  - Query result cache to skip LLM on repeated questions
  - Post-execution: normalization, insight, viz, lineage, audit (Steps 6–11)
"""
import os
import json
import hashlib
import time

from langchain_core.runnables import RunnableParallel, RunnableLambda

from config import csm, glossary, decomposition_chain, llm
from core.retriever_context import retrieve_once, build_rag_context, extract_rag_hints
from core.intent_parser import IntentParser
from query_engine.graph_resolver import rag_plus_plus_resolver
from query_engine.sql_compiler import sql_compiler
from core.cost_estimator import estimate_pipeline_cost, format_cost

# ── One-time startup work ────────────────────────────────────────────────────

intent_parser = IntentParser(csm, glossary)

# ── Helper functions ─────────────────────────────────────────────────────────

def _get_model_name(llm) -> str:
    """
    Safely extracts model name from any LangChain LLM object.
    Different providers use different attribute names.
    """
    for attr in ("model_name", "model", "model_id"):
        val = getattr(llm, attr, None)
        if val and isinstance(val, str):
            return val
    # Last resort — check pydantic fields
    fields = getattr(llm, "__fields__", {})
    for attr in ("model_name", "model"):
        if attr in fields:
            return getattr(llm, attr, "unknown")
    return "unknown"

# Schema context is static — build it once at import time, never again
_SCHEMA_CONTEXT = json.dumps({
    "metrics": {
        k: {"sources": v.get("sources", []), "label": v.get("label", "")}
        for k, v in csm.get("metrics", {}).items()
    },
    "dimensions": {
        k: {"source": v.get("source", ""), "type": v.get("type", "")}
        for k, v in csm.get("dimensions", {}).items()
    },
})

# ── Simple SQL result cache ──────────────────────────────────────────────────

_CACHE: dict = {}
_CACHE_TTL   = 60 * 60 * 8   # 8 hours

def _cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]

def cache_get(question: str) -> dict | None:
    entry = _CACHE.get(_cache_key(question))
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        print("  [Cache] HIT — skipping LLM and RAG")
        return entry["data"]
    return None

def cache_put(question: str, data: dict):
    _CACHE[_cache_key(question)] = {"ts": time.time(), "data": data}
    if len(_CACHE) > 500:
        # Evict oldest 100
        oldest = sorted(_CACHE.items(), key=lambda x: x[1]["ts"])[:100]
        for k, _ in oldest:
            del _CACHE[k]

# ── Pipeline steps ───────────────────────────────────────────────────────────

def _prepare_context(question: str) -> dict:
    """ Step 1: Single retrieve() call. """
    rag_result = retrieve_once(question)
    
    # Extract the technical strings for the prompt
    # This uses the public API from rag_retriever.py [cite: 70]
    from core.rag_retriever import build_retrieved_context
    
    metrics_list_str = ", ".join(sorted(csm.get("metrics", {}).keys()))
    
    # RAG Confidence Fallback
    q_lower = question.lower()
    _RANKING_SIGNALS = [
        "top", "most", "highest", "lowest", "best", "expensive",
        "cheapest", "worst", "priciest", "costliest",
    ]
    is_ranking = any(kw in q_lower for kw in _RANKING_SIGNALS)
    confidence = getattr(rag_result, "metric_confidence", "none")
    
    schema_suffix = ""
    if is_ranking and confidence in ("low", "none"):
        schema_suffix = "\n\nCRITICAL HINT: This is a ranking question. Only choose a metric if its 'usage_hint' says it is for ranking/scalar values. DO NOT choose aggregate metrics for ranking questions."
        scalar_keys = [k for k, v in csm.get("metrics", {}).items() if v.get("compute_type") == "scalar"]
        if scalar_keys:
            metrics_list_str = ", ".join(sorted(scalar_keys))
            
    state = {
        "question": question,
        "schema_context": _SCHEMA_CONTEXT + schema_suffix,
        "bgo_context": build_rag_context(rag_result, glossary),
        # Issue 7: LLM must see ALL valid CSM keys, not just RAG results,
        # so it can produce exact metric keys instead of hallucinated ones.
        "metrics_list": metrics_list_str,
        "dimensions_list": ", ".join(sorted(csm.get("dimensions", {}).keys())),
        "_rag_result": rag_result,
    }

    # Step 1.5: Estimate cost before decomposition LLM call
    cost_estimate = estimate_pipeline_cost(
        question       = question,
        schema_context = state["schema_context"],
        bgo_context    = state["bgo_context"],
        model          = _get_model_name(llm),
    )
    print(f"\n[Cost] {format_cost(cost_estimate)}")
    state["_cost_estimate"] = cost_estimate

    return state


def _parse_intent(state: dict) -> dict:
    """
    Step 3: Reuses the already-fetched rag_result from state.
    Runs the full IntentParser resolution pipeline.
    """
    raw_intent = state["intent"]
    question   = state["question"]
    rag_result = state["_rag_result"]

    rag_hints      = extract_rag_hints(rag_result)
    resolved_state = intent_parser.process_intent(raw_intent, question, rag_hints)

    return {
        **state,
        "intent":   resolved_state["intent"],
    }


def _ensure_metric_exists(state: dict) -> dict:
    """
    Step 3.5: Checks if the metric exists in the CSM. 
    If not, attempts to auto-generate it.
    If generation fails or the resulting SQL is invalid, falls back to interactive CLI prompt.
    NOTE: The input() fallback here is CLI-only. If this pipeline is ever served over HTTP,
    this must be replaced with a callback/webhook or exception-based flow.
    """
    intent = state["intent"]
    metric_key = intent.get("metric")
    import config
    
    if (metric_key and metric_key not in config.csm.get("metrics", {})) or intent.get("_metric_type_mismatch"):
        # If type mismatch, try to find a scalar sibling
        if intent.get("_metric_type_mismatch"):
            metric_node = config.csm.get("metrics", {}).get(metric_key, {})
            source = metric_node.get("sources", [None])[0]
            label_tokens = set(metric_node.get("label", "").lower().split())
            
            for key, node in config.csm.get("metrics", {}).items():
                if node.get("compute_type") == "scalar":
                    sibling_source = node.get("sources", [None])[0]
                    sibling_label_tokens = set(node.get("label", "").lower().split())
                    
                    if sibling_source == source or (label_tokens & sibling_label_tokens):
                        print(f"  [Pipeline] Resolved type mismatch: swapped '{metric_key}' -> '{key}'")
                        intent["metric"] = key
                        intent.pop("_metric_type_mismatch", None)
                        return state
                        
        from core.metric_generator import auto_generate_metric, write_metric_to_yaml
        from query_engine.graph_resolver import rag_plus_plus_resolver
        from query_engine.sql_compiler import sql_compiler
        from db import execute_query
        import sys
        
        success = auto_generate_metric(metric_key, state["question"], intent)
        
        def test_pipeline():
            plan = rag_plus_plus_resolver(intent)
            sql, params = sql_compiler(plan)
            # Validate SQL with LIMIT 0 to catch db errors
            test_sql = sql + " LIMIT 0" if "LIMIT" not in sql.upper() else sql
            execute_query(test_sql, params)
            return True
            
        is_valid = False
        if success:
            try:
                is_valid = test_pipeline()
            except Exception as e:
                print(f"  [AutoGenerate] Validation failed: {e}")
                is_valid = False
                
        if not is_valid:
            print(f"\n⚠️  I could not automatically define the metric \"{metric_key}\".")
            print("Please provide the following details (press Enter to skip optional fields):\n")
            compute_expr = input("  Compute expression (e.g. COUNT(DISTINCT rental.rental_id)): ").strip()
            source_tables = input("  Source tables, comma-separated (e.g. rental, inventory, film): ").strip()
            join_path = input("  Join path in order, comma-separated (optional): ").strip()
            label = input("  Label (human-readable name): ").strip()
            
            if not compute_expr or not source_tables:
                print("Sorry, I need at least a compute expression and source tables to proceed.")
                sys.exit(0)
                
            metric_data = {
                "compute": compute_expr,
                "sources": [s.strip() for s in source_tables.split(',')],
                "label": label or metric_key
            }
            if join_path:
                metric_data["join_path"] = [j.strip() for j in join_path.split(',')]
                
            write_metric_to_yaml(metric_key, metric_data, state["question"])
            
            from build_embeddings import build_embeddings_if_stale
            from core.rag_retriever import _IndexStore
            build_embeddings_if_stale(force=True)
            _IndexStore._instance = None
            
            try:
                test_pipeline()
            except Exception as e:
                print(f"Sorry, the manually provided metric resulted in an error: {e}")
                sys.exit(0)
                
    return state


def _resolve_joins(state: dict) -> dict:
    """Step 4: Graph-based join resolution → logical plan."""
    return {
        **state,
        "logical_plan": rag_plus_plus_resolver(state["intent"]),
    }


def _compile_sql(state: dict) -> dict:
    """Step 5: Logical plan → SQL string + params."""
    plan = state["logical_plan"]
    sql, params = sql_compiler(plan)
    print(f"\nGenerated SQL:\n{sql}\n")
    if params:
        print(f"  [Params]: {params}")
    return {
        **state,
        "sql":          sql,
        "sql_params":   params,
        "logical_plan": plan,
    }


def _execute_sql(state: dict) -> dict:
    """Step 6: Execute SQL against database with parameterized values."""
    from db import execute_query
    try:
        rows = execute_query(state["sql"], state.get("sql_params"))
        return {**state, "raw_rows": rows}
    except Exception as e:
        print(f"  [ExecuteSQL] Database error: {e}")
        return {**state, "raw_rows": [], "execution_error": str(e)}


def _normalize_results(state: dict) -> dict:
    """Step 7: Normalize raw rows into Dataset."""
    from core.result_normalizer import normalize

    if state.get("execution_error"):
        # DB failed — produce an empty Dataset and pass error forward
        dataset = {
            "columns": [],
            "rows": [],
            "stats": {},
            "row_count": 0,
            "display_hint": "empty",
        }
        return {**state, "results": dataset}

    dataset = normalize(state["raw_rows"])
    return {**state, "results": dataset}


def _generate_insight(state: dict) -> dict:
    """Step 8: Generate LLM insight from stats."""
    from core.insight_engine import generate_insight
    insight = generate_insight(state["results"], state.get("question", ""))
    
    # Step 8.5: Estimate insight generation cost
    from core.cost_estimator import TokenCounter, CostModel, CostEstimate
    insight_prompt = str(state.get("results", ""))
    model_name = _get_model_name(llm)
    counter  = TokenCounter(model_name)
    est      = counter.estimate(insight_prompt, 150)
    insight_cost = CostModel(model_name).calculate(est)

    # Accumulate totals
    prior = state.get("_cost_estimate")
    if prior:
        combined = CostEstimate(
            model             = prior.model,
            prompt_tokens     = prior.prompt_tokens + insight_cost.prompt_tokens,
            completion_tokens = prior.completion_tokens + insight_cost.completion_tokens,
            total_tokens      = prior.total_tokens + insight_cost.total_tokens,
            input_cost_usd    = prior.input_cost_usd + insight_cost.input_cost_usd,
            output_cost_usd   = prior.output_cost_usd + insight_cost.output_cost_usd,
            total_cost_usd    = prior.total_cost_usd + insight_cost.total_cost_usd,
            estimated         = prior.estimated or insight_cost.estimated,
        )
        print(f"  [Cost] Insight step: {format_cost(insight_cost)}")
        print(f"  [Cost] Session total: {format_cost(combined)}")
        state["_cost_estimate"] = combined

    return {**state, "insight": insight}


def _recommend_viz(state: dict) -> dict:
    """Step 9: Recommend visualization."""
    from core.viz_recommender import recommend_viz
    spec = recommend_viz(state["results"])
    return {**state, "viz_spec": spec}


def _build_lineage_step(state: dict) -> dict:
    """Step 10: Build lineage record."""
    from core.lineage_tracker import build_lineage
    lineage = build_lineage(state)
    return {**state, "lineage": lineage}


def _log_audit(state: dict) -> dict:
    """Step 11: Write audit log entry."""
    from core.audit_logger import log_run
    run_id = log_run(state)
    return {**state, "audit_id": run_id}


# ── Assembled pipeline ───────────────────────────────────────────────────────

analytics_pipeline = (
    # Step 1: Retrieve once, build all context
    RunnableLambda(_prepare_context)

    # Step 2: LLM decomposition — Text → draft JSON intent
    | RunnableParallel({
        "intent":         decomposition_chain,
        "question":       lambda x: x["question"],
        "_rag_result":    lambda x: x["_rag_result"],
        "_cost_estimate": lambda x: x["_cost_estimate"],
    })

    # Step 3: Intent parsing and conflict resolution
    | RunnableLambda(_parse_intent)

    # Step 3.5: Ensure metric exists (auto-generate or fallback)
    | RunnableLambda(_ensure_metric_exists)

    # Step 4: Graph-based join resolution
    | RunnableLambda(_resolve_joins)

    # Step 5: SQL compilation
    | RunnableLambda(_compile_sql)

    # Step 6: SQL execution
    | RunnableLambda(_execute_sql)

    # Step 7: Result normalization
    | RunnableLambda(_normalize_results)

    # Step 8: Insight generation
    | RunnableLambda(_generate_insight)

    # Step 9: Visualization recommendation
    | RunnableLambda(_recommend_viz)

    # Step 10: Lineage tracking
    | RunnableLambda(_build_lineage_step)

    # Step 11: Audit logging
    | RunnableLambda(_log_audit)
)