"""
connector.py  (AXON edition)
----------------------------
Replaces the flat-CSM connector with one backed by axon_compiler.py.

What changed vs the original:
  - CSM + BGO YAML replaced by a single axon.yaml
  - sql_compiler() replaced by AxonCompiler.compile()
  - Intent now carries: metric | recipe, dimensions, filters, sort, limit, mode
  - Supports ratio / window / composite / subquery / filtered_dim / recipe queries
  - LLM prompt is auto-generated from axon.yaml synonyms → no stale bgo.yaml drift
  - enforce_ranking() and normalize_intent() are retained (simplified)

Usage:
    python connector.py
    > Ask your question: who are the top 5 most efficient employees?
"""

from __future__ import annotations

import decimal
import json

from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

from axon_compiler import AxonCompiler


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DB_URL       = "mysql+pymysql://root:@localhost:3306/test"
OLLAMA_MODEL = "llama3"

engine   = create_engine(DB_URL, future=True)
compiler = AxonCompiler("axon.yaml")
axon     = compiler.axon


# ═══════════════════════════════════════════════════════════════════════════
# LLM DECOMPOSITION PROMPT  (auto-built from axon.yaml)
# ═══════════════════════════════════════════════════════════════════════════

def _build_prompt_template() -> str:
    ctx = compiler.get_prompt_context()

    recipe_examples = "\n".join(
        f'  recipe "{k}": triggers: {", ".join(v.get("synonyms", [])[:4])}'
        for k, v in axon.recipes.items()
    )

    measure_examples = "\n".join(
        f'  measure "{k}": triggers: {", ".join(v.get("synonyms", [])[:4])}'
        for k, v in axon.measures.items()
    )

    dim_examples = "\n".join(
        f'  dimension "{k}": triggers: {", ".join(v.get("synonyms", [])[:3])}'
        for k, v in axon.dimensions.items()
    )

    template = f"""You are a precise semantic parser for a business analytics system.
Translate the user's question into a JSON intent object.

AVAILABLE MEASURES (use key exactly as written):
{", ".join(ctx["measure_keys"])}

AVAILABLE DIMENSIONS (use key exactly as written):
{", ".join(ctx["dimension_keys"])}

AVAILABLE RECIPES (use key exactly as written):
{", ".join(ctx["recipe_keys"])}

SYNONYM GUIDE:
{ctx["synonym_guide"]}

RECIPES (use these for multi-step / report queries):
{recipe_examples}

MEASURE TRIGGERS:
{measure_examples}

DIMENSION TRIGGERS:
{dim_examples}

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
{{
  "intent": {{
    "recipe":     "<recipe_key or null>",
    "metric":     "<measure_key or null>",
    "dimensions": ["<dimension_key>", ...],
    "filters":    [{{"field": "<dimension_key>", "operator": "equals|contains|gt|gte|lt|lte|in|notIn|last_n_days", "values": ["<value>"]}}],
    "sort":       "asc|desc|null",
    "limit":      <integer or null>,
    "mode":       "list|null"
  }}
}}

RULES:
1. If the question is a report / multi-step / ranking across all employees or departments, prefer a RECIPE.
2. Use recipe OR metric — not both (recipe takes priority).
3. mode="list" when the question uses: show, list, display, get, fetch, all, every.
4. sort="desc" + limit=1 for: most, top, best, highest, largest, maximum.
5. sort="asc"  + limit=1 for: least, lowest, fewest, smallest, minimum.
6. dimensions: group-by fields. Only add when the user asks "by/per/for each <entity>".
7. filters: ONLY when a specific value is mentioned (e.g. "in Engineering", "status = done").
8. Return null (not the string "null") for unused fields.

EXAMPLES:

Q: "list all employees"
A: {{"intent": {{"recipe": null, "metric": "total_employees", "dimensions": ["employee_name", "employee_id"], "filters": [], "sort": null, "limit": null, "mode": "list"}}}}

Q: "how many tasks does each department have?"
A: {{"intent": {{"recipe": null, "metric": "total_tasks", "dimensions": ["department_name"], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "who are the top 3 most efficient employees?"
A: {{"intent": {{"recipe": null, "metric": "task_completion_rate", "dimensions": ["employee_name"], "filters": [], "sort": "desc", "limit": 3, "mode": null}}}}

Q: "show me completed tasks for each project"
A: {{"intent": {{"recipe": null, "metric": "tasks_done", "dimensions": ["project_name"], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "which department has the best performance?"
A: {{"intent": {{"recipe": "department_health_report", "metric": null, "dimensions": [], "filters": [], "sort": "desc", "limit": 1, "mode": null}}}}

Q: "give me an efficiency report for all employees"
A: {{"intent": {{"recipe": "employee_efficiency_report", "metric": null, "dimensions": [], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "who manages the most people?"
A: {{"intent": {{"recipe": "manager_span_of_control", "metric": null, "dimensions": [], "filters": [], "sort": "desc", "limit": 1, "mode": null}}}}

Q: "rank employees by completion rate within each department"
A: {{"intent": {{"recipe": "top_performers_by_department", "metric": null, "dimensions": [], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "how many blocked tasks does the Engineering department have?"
A: {{"intent": {{"recipe": null, "metric": "tasks_blocked", "dimensions": ["department_name"], "filters": [{{"field": "department_name", "operator": "equals", "values": ["Engineering"]}}], "sort": null, "limit": null, "mode": null}}}}

Q: "show tasks completed in the last 30 days"
A: {{"intent": {{"recipe": null, "metric": "tasks_completed_last_30_days", "dimensions": [], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "which employees have above average workload?"
A: {{"intent": {{"recipe": null, "metric": "employees_above_avg_tasks", "dimensions": [], "filters": [], "sort": null, "limit": null, "mode": null}}}}

Q: "overall performance score for each employee"
A: {{"intent": {{"recipe": null, "metric": "employee_performance_score", "dimensions": ["employee_name"], "filters": [], "sort": "desc", "limit": null, "mode": null}}}}

QUESTION: {{question}}
RESPONSE:"""

    return template


_PROMPT_TEMPLATE = _build_prompt_template()

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")

decomposition_chain = (
    ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    | llm
    | JsonOutputParser()
)


# ═══════════════════════════════════════════════════════════════════════════
# RANKING ENFORCER
# ═══════════════════════════════════════════════════════════════════════════

RANKING_KEYWORDS = {
    "desc": ["most", "highest", "top", "best", "costliest", "largest",
             "maximum", "max", "biggest", "best performing"],
    "asc":  ["least", "lowest", "cheapest", "smallest", "minimum", "min",
             "fewest", "worst"],
}


def enforce_ranking(intent_output: dict, question: str) -> dict:
    data = intent_output.get("intent", intent_output)
    q    = question.lower()
    for direction, keywords in RANKING_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            if not data.get("sort"):
                data["sort"] = direction
                print(f"  [enforcer] sort={direction}")
            break
    if "intent" in intent_output:
        intent_output["intent"] = data
    else:
        intent_output = data
    return intent_output


# ═══════════════════════════════════════════════════════════════════════════
# INTENT NORMALISER
# ═══════════════════════════════════════════════════════════════════════════

LIST_TRIGGERS = {
    "list", "show", "display", "all", "every", "get", "fetch",
    "give me", "who are", "what are", "tell me", "enumerate",
}


def normalize_intent(intent_output: dict, question: str = "") -> dict:
    """
    Normalise LLM output:
    1. Resolve measure/dimension/recipe keys case-insensitively via AxonLoader.
    2. Flatten filter value lists.
    3. Detect list intent from question keywords.
    """
    data = intent_output.get("intent", intent_output)

    # ── Recipe key ────────────────────────────────────────────────────────
    raw_recipe = data.get("recipe") or ""
    if raw_recipe:
        rk, _ = axon.resolve_recipe(str(raw_recipe))
        if rk:
            data["recipe"] = rk
        else:
            print(f"  [warn] Recipe '{raw_recipe}' not found — cleared")
            data["recipe"] = None

    # ── Measure key ───────────────────────────────────────────────────────
    raw_metric = data.get("metric") or ""
    if raw_metric:
        mk, _ = axon.resolve_measure(str(raw_metric))
        if mk:
            data["metric"] = mk
        else:
            print(f"  [warn] Measure '{raw_metric}' not found — cleared")
            data["metric"] = None

    # ── Dimension keys ────────────────────────────────────────────────────
    resolved_dims = []
    for raw_dim in data.get("dimensions", []):
        dk, _ = axon.resolve_dimension(str(raw_dim))
        if dk:
            resolved_dims.append(dk)
        else:
            print(f"  [warn] Dimension '{raw_dim}' not found — skipped")
    data["dimensions"] = resolved_dims

    # ── Filter keys + values ──────────────────────────────────────────────
    normalised_filters = []
    for f in data.get("filters", []):
        raw_key = f.get("col_key") or f.get("field") or ""
        dk, _   = axon.resolve_dimension(str(raw_key))
        if not dk:
            print(f"  [warn] Filter field '{raw_key}' not found — skipped")
            continue
        values = (
            f.get("values") or f.get("vals")
            or ([f.get("val")] if f.get("val") is not None else [])
        )
        if values and isinstance(values, list) and values and isinstance(values[0], list):
            values = [item for sub in values for item in sub]
        values = [v for v in values if v is not None and str(v).strip() != ""]
        if not values:
            continue
        normalised_filters.append({
            "col_key":  dk,
            "field":    dk,
            "val":      values[0],
            "vals":     values,
            "values":   values,
            "operator": f.get("operator") or f.get("op") or "equals",
            "op":       f.get("operator") or f.get("op") or "equals",
        })
        if len(values) > 1:
            print(f"  [normalizer] multi-value filter: {dk} IN {values}")
    data["filters"] = normalised_filters

    # ── Detect list intent from question ──────────────────────────────────
    q              = question.lower()
    is_list_q      = any(kw in q for kw in LIST_TRIGGERS)
    metric_is_count = (data.get("metric") or "").endswith("_row_count")
    has_no_sort    = not data.get("sort")
    llm_said_list  = data.get("mode") == "list"

    if (is_list_q and metric_is_count and has_no_sort) or llm_said_list:
        data["mode"] = "list"
        print("  [normalizer] mode=list")
        # Auto-inject id + name dims if none were found
        if not data["dimensions"] and data.get("metric"):
            mk = data["metric"]
            mn = axon.measures.get(mk, {})
            base = mn.get("source", "")
            if base:
                id_dim   = f"{base}_id"
                name_dim = next(
                    (k for k, v in axon.dimensions.items()
                     if v.get("source") == base and v.get("type") == "string"),
                    None,
                )
                auto = [d for d in [id_dim, name_dim] if d and d in axon.dimensions]
                if auto:
                    data["dimensions"] = auto
                    print(f"  [normalizer] auto-injected dims: {auto}")

    if "intent" in intent_output:
        intent_output["intent"] = data
    else:
        intent_output = data
    return intent_output


# ═══════════════════════════════════════════════════════════════════════════
# AXON COMPILE STEP (replaces rag_plus_plus_resolver + sql_compiler)
# ═══════════════════════════════════════════════════════════════════════════

def axon_compile(intent_output: dict) -> dict:
    """
    Takes normalised intent, compiles via AxonCompiler, returns sql + plan.
    """
    data = intent_output.get("intent", intent_output)

    # Build intent dict for compiler (unified shape)
    intent = {
        "recipe":     data.get("recipe"),
        "metric":     data.get("metric"),
        "dimensions": data.get("dimensions", []),
        "filters":    data.get("filters", []),
        "sort":       data.get("sort"),
        "limit":      data.get("limit"),
        "mode":       data.get("mode"),
    }

    sql, plan = compiler.compile(intent)
    return {"sql": sql, "plan": plan, "intent": data}


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

analytics_pipeline = (
    RunnableParallel({
        "question": RunnablePassthrough(),
    })
    | RunnableParallel({
        "intent":   decomposition_chain,
        "question": lambda x: x["question"],
    })
    | RunnableLambda(lambda x: {
        "intent":   enforce_ranking(x["intent"], x["question"]),
        "question": x["question"],
    })
    | RunnableLambda(lambda x: {
        "intent":   normalize_intent(x["intent"], x["question"]),
        "question": x["question"],
    })
    | RunnableLambda(lambda x: axon_compile(x["intent"]))
)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


def ask_database(question: str) -> list[dict]:
    print(f"\nProcessing: {question}")
    print("Running pipeline...")

    output   = analytics_pipeline.invoke(question)
    final_sql = output["sql"]
    plan      = output["plan"]

    print(f"\nPlan type : {plan.plan_type}")
    print(f"Generated SQL:\n{final_sql}\n")

    with engine.connect() as conn:
        result = conn.execute(text(final_sql))
        return [dict(row._mapping) for row in result]


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("AXON Analytics Connector")
    print("=" * 50)
    print("Supported query types:")
    print("  • Simple lists (show/list/get)")
    print("  • Aggregates (count/sum per dimension)")
    print("  • Ratios (completion rate, block rate)")
    print("  • Window functions (rank employees globally or within dept)")
    print("  • Composite scores (weighted performance score)")
    print("  • Subquery thresholds (above-average workload)")
    print("  • Recipes (multi-CTE efficiency/dept/manager reports)")
    print("=" * 50)

    while True:
        try:
            question = input("\nAsk your question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        try:
            results = ask_database(question)
            print("Results:", json.dumps(results, indent=2, default=decimal_default))
        except Exception as e:
            print(f"Error: {e}")