import yaml
import json
import decimal
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough


yaml.safe_load(open('csm_enterprise.yaml'))
yaml.safe_load(open('bgo.yaml'))

with open('bgo.yaml', 'r') as f:
    glossary = yaml.safe_load(f)
with open('csm_enterprise.yaml', 'r') as f:
    csm = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Case-insensitive CSM key resolver
# ---------------------------------------------------------------------------

_METRIC_KEY_MAP    = {k.lower(): k for k in csm.get('metrics', {}).keys()}
_DIMENSION_KEY_MAP = {k.lower(): k for k in csm.get('dimensions', {}).keys()}


def resolve_metric_key(raw: str) -> str | None:
    return _METRIC_KEY_MAP.get(raw.lower()) if raw else None


def resolve_dimension_key(raw: str) -> str | None:
    return _DIMENSION_KEY_MAP.get(raw.lower()) if raw else None


# ---------------------------------------------------------------------------
# BGO context builder
# ---------------------------------------------------------------------------

def build_bgo_context(glossary: dict) -> str:
    lines = []

    lines.append("## METRIC SYNONYMS")
    lines.append("If the user's phrasing matches any synonym, use that metric key exactly.\n")
    for metric_key, synonyms in glossary.get("metrics", {}).items():
        lines.append(f"  {metric_key}:")
        lines.append(f"    triggers: {', '.join(synonyms)}")
    lines.append("")

    lines.append("## DIMENSION SYNONYMS")
    lines.append("If the user's phrasing matches any synonym, use that dimension key exactly.\n")
    for dim_key, synonyms in glossary.get("dimensions", {}).items():
        lines.append(f"  {dim_key}:")
        lines.append(f"    triggers: {', '.join(synonyms)}")
    lines.append("")

    lines.append("## ENTITY DEFINITIONS")
    for entity, meta in glossary.get("ontology", {}).get("entities", {}).items():
        syns = ", ".join(meta.get("synonyms", []))
        lines.append(f"  {entity}: {meta['description']}")
        lines.append(f"    also called: {syns}")
    lines.append("")

    lines.append("## ENTITY RELATIONSHIPS")
    lines.append("Use these to determine which dimension to group by.\n")
    for rel in glossary.get("ontology", {}).get("relationships", []):
        nl = "; ".join(rel.get("natural_language", []))
        lines.append(f"  {rel['statement']}")
        lines.append(f"    natural language triggers: {nl}")
    lines.append("")

    lines.append("## INTENT PATTERNS")
    lines.append("Concrete mappings from question shape to metric + dimensions.\n")
    for p in glossary.get("intent_patterns", []):
        lines.append(f"  pattern : \"{p['pattern']}\"")
        lines.append(f"  metric  : {p['metric']}")
        lines.append(f"  dims    : {p.get('dimensions', [])}")
        if p.get("filters"):
            lines.append(f"  filters : {p['filters']}")
        if p.get("sort"):
            lines.append(f"  sort    : {p['sort']}, limit: {p.get('limit', 1)}")
        if p.get("mode"):
            lines.append(f"  mode    : {p['mode']}")
        lines.append("")

    return "\n".join(lines)


engine = create_engine(
    "mysql+pymysql://root:@localhost:3306/test",
    future=True,
)

llm = ChatOllama(model="llama3", temperature=0, format="json")

with open("decomposition_prompt.txt", encoding="utf-8") as _f:
    _PROMPT_TEMPLATE = _f.read()

decomposition_chain = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE) | llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# WHERE clause helper
# ---------------------------------------------------------------------------

def _build_where_clauses(filters: list) -> list[str]:
    clauses = []
    for f in filters:
        if f.get("is_aggregate"):
            continue

        col_key  = f.get("col_key", "")
        dim_node = csm['dimensions'].get(col_key)
        if not dim_node:
            print(f"  [warn] Filter key '{col_key}' not in CSM dimensions")
            continue

        col     = dim_node['column']
        dtype   = dim_node.get('type', 'string')
        col_ref = col if dim_node.get('is_time') else f"{dim_node['source']}.{col}"

        raw_val  = f.get('val')
        raw_vals = f.get('vals')
        if raw_vals and isinstance(raw_vals, list) and len(raw_vals) > 1:
            values = raw_vals
        else:
            values = [raw_val] if raw_val is not None else []

        if not values:
            continue

        op = f.get('op', 'equals')

        if dtype in ('id', 'number'):
            try:
                num_vals = [int(v) if str(v).isdigit() else float(v) for v in values]
            except (ValueError, TypeError):
                num_vals = values

            if len(num_vals) == 1:
                if op in ('gt', '>'):
                    clauses.append(f"{col_ref} > {num_vals[0]}")
                elif op in ('gte', '>='):
                    clauses.append(f"{col_ref} >= {num_vals[0]}")
                elif op in ('lt', '<'):
                    clauses.append(f"{col_ref} < {num_vals[0]}")
                elif op in ('lte', '<='):
                    clauses.append(f"{col_ref} <= {num_vals[0]}")
                elif op == 'notEquals':
                    clauses.append(f"{col_ref} != {num_vals[0]}")
                else:
                    clauses.append(f"{col_ref} = {num_vals[0]}")
            else:
                in_list = ", ".join(str(v) for v in num_vals)
                clauses.append(f"{col_ref} IN ({in_list})")

        elif dtype == 'time':
            safe_val = str(values[0]).replace("'", "''")
            if op in ('gt', '>'):
                clauses.append(f"{col_ref} > '{safe_val}'")
            elif op in ('gte', '>='):
                clauses.append(f"{col_ref} >= '{safe_val}'")
            elif op in ('lt', '<'):
                clauses.append(f"{col_ref} < '{safe_val}'")
            elif op in ('lte', '<='):
                clauses.append(f"{col_ref} <= '{safe_val}'")
            else:
                clauses.append(f"{col_ref} = '{safe_val}'")

        else:
            safe_vals = [str(v).replace("'", "''") for v in values]
            if len(safe_vals) > 1:
                in_parts = ', '.join(f"LOWER('{v}')" for v in safe_vals)
                clauses.append(f"LOWER({col_ref}) IN ({in_parts})")
            elif op == 'contains':
                clauses.append(f"LOWER({col_ref}) LIKE LOWER('%{safe_vals[0]}%')")
            elif op == 'notEquals':
                clauses.append(f"LOWER({col_ref}) != LOWER('{safe_vals[0]}')")
            else:
                clauses.append(f"LOWER({col_ref}) = LOWER('{safe_vals[0]}')")

    return clauses


# ---------------------------------------------------------------------------
# SQL compiler
# ---------------------------------------------------------------------------

def sql_compiler(plan):
    if plan.get("mode") == "list":
        cols = plan.get("select_cols")
        if not cols:
            cols = [f"{plan['from']}.*"]
        sql  = f"SELECT {', '.join(cols)}"
        sql += f"\nFROM {plan['from']}"

        if plan.get("joins"):
            sql += "\n" + "\n".join(plan["joins"])

        where_clauses = _build_where_clauses(plan.get("filters", []))
        if where_clauses:
            sql += f"\nWHERE {' AND '.join(where_clauses)}"

        if plan.get("sort") == "desc":
            sql += f"\nORDER BY {cols[-1]} DESC"
        elif plan.get("sort") == "asc":
            sql += f"\nORDER BY {cols[-1]} ASC"

        if plan.get("limit"):
            sql += f"\nLIMIT {plan['limit']}"

        return sql

    # Aggregate mode
    select_items = plan.get('select', [])
    if plan.get('group_by'):
        sql = f"SELECT {', '.join(select_items)}"
    else:
        metric_part = [s for s in select_items if 'AS result' in s]
        sql = f"SELECT {', '.join(metric_part)}"

    sql += f"\nFROM {plan['from']}"

    if plan.get('joins'):
        sql += "\n" + "\n".join(plan['joins'])

    where_filters = [f for f in plan.get('filters', []) if not f.get('is_aggregate')]
    clauses = _build_where_clauses(where_filters)
    if clauses:
        sql += f"\nWHERE {' AND '.join(clauses)}"

    if plan.get('group_by'):
        sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    having_filters = [f for f in plan.get('filters', []) if f.get('is_aggregate')]
    if having_filters:
        having_clauses = []
        for f in having_filters:
            metric = csm['metrics'].get(f['col_key'])
            if metric:
                having_clauses.append(f"{metric['compute']} {f['op']} {f['val']}")
        if having_clauses:
            sql += f"\nHAVING {' AND '.join(having_clauses)}"

    if plan.get('sort') == 'desc':
        sql += "\nORDER BY result DESC"
    elif plan.get('sort') == 'asc':
        sql += "\nORDER BY result ASC"

    if plan.get('limit'):
        sql += f"\nLIMIT {plan['limit']}"

    return sql


# ---------------------------------------------------------------------------
# List-intent keywords
# ---------------------------------------------------------------------------

LIST_TRIGGERS = {
    "list", "show", "display", "all", "every", "get", "fetch",
    "give me", "who are", "what are", "tell me", "enumerate",
}

RANKING_KEYWORDS = {
    "desc": ["most", "highest", "top", "best", "costliest", "expensive",
             "largest", "maximum", "max", "biggest"],
    "asc":  ["least", "lowest", "cheapest", "smallest", "minimum", "min", "fewest"],
}


def enforce_ranking(intent_output: dict, question: str) -> dict:
    data = intent_output.get('intent', intent_output)
    q    = question.lower()

    for direction, keywords in RANKING_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            if not data.get('sort'):
                data['sort'] = direction
                print(f"  [enforcer] sort={direction} (keyword match)")
            if not data.get('limit'):
                data['limit'] = 1
                print(f"  [enforcer] limit=1 (keyword match)")
            break

    if 'intent' in intent_output:
        intent_output['intent'] = data
    else:
        intent_output = data

    return intent_output


# ---------------------------------------------------------------------------
# Status keyword enforcer
# LLMs (especially small local models) frequently drop multi-value filters.
# This step scans the raw question for task-status vocabulary and injects
# a tasks_task_status filter when the LLM missed it or only captured one value.
# ---------------------------------------------------------------------------

# Maps natural-language words -> canonical DB values in tasks.task_status
_STATUS_KEYWORD_MAP = {
    "completed":    "done",
    "complete":     "done",
    "done":         "done",
    "finished":     "done",
    "closed":       "done",
    "in progress":  "in_progress",
    "in_progress":  "in_progress",
    "inprogress":   "in_progress",
    "ongoing":      "in_progress",
    "active":       "in_progress",
    "started":      "in_progress",
    "working":      "in_progress",
    "pending":      "pending",
    "open":         "pending",
    "not started":  "pending",
    "waiting":      "pending",
    "blocked":      "blocked",
    "stuck":        "blocked",
}


def enforce_status_filter(intent_output: dict, question: str) -> dict:
    """
    Detect task-status keywords in the question and ensure the filter list
    contains a correctly formed tasks_task_status filter with ALL mentioned
    statuses.  Overwrites any partial/incorrect filter the LLM emitted.
    """
    data = intent_output.get('intent', intent_output)
    q    = question.lower()

    # Collect every canonical status value mentioned in the question.
    # Use longest-match first so "in progress" beats "progress".
    matched_statuses: list[str] = []
    for phrase in sorted(_STATUS_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q and _STATUS_KEYWORD_MAP[phrase] not in matched_statuses:
            matched_statuses.append(_STATUS_KEYWORD_MAP[phrase])

    if not matched_statuses:
        # No status words found — leave filters untouched
        if 'intent' in intent_output:
            intent_output['intent'] = data
        else:
            intent_output = data
        return intent_output

    print(f"  [status-enforcer] detected statuses in question: {matched_statuses}")

    # Remove any existing (possibly wrong/incomplete) tasks_task_status filter
    existing = [
        f for f in data.get('filters', [])
        if f.get('col_key') != 'tasks_task_status'
    ]

    # Build a properly normalised filter entry
    status_filter = {
        "col_key":      "tasks_task_status",
        "val":          matched_statuses[0],       # primary value
        "vals":         matched_statuses,           # full list for IN(...)
        "op":           "equals",
        "is_aggregate": False,
    }
    data['filters'] = existing + [status_filter]
    print(f"  [status-enforcer] injected filter: tasks_task_status IN {matched_statuses}")

    if 'intent' in intent_output:
        intent_output['intent'] = data
    else:
        intent_output = data

    return intent_output


# ---------------------------------------------------------------------------
# Intent normaliser
# ---------------------------------------------------------------------------

def normalize_intent(intent_output: dict, question: str = "") -> dict:
    data = intent_output.get('intent', intent_output)

    # Normalise metric key
    raw_metric       = data.get("metric", "")
    canonical_metric = resolve_metric_key(raw_metric)
    if canonical_metric:
        data["metric"] = canonical_metric
    elif raw_metric:
        print(f"  [warn] Metric '{raw_metric}' not in CSM -- keeping as-is")

    # Normalise dimension keys
    resolved_dims = []
    for raw_dim in data.get("dimensions", []):
        canonical = resolve_dimension_key(raw_dim)
        if canonical:
            resolved_dims.append(canonical)
        else:
            print(f"  [warn] Dimension '{raw_dim}' not in CSM -- skipped")
    data["dimensions"] = resolved_dims

    # Normalise filter keys + values
    normalized_filters = []
    for f in data.get("filters", []):
        raw_key = f.get("col_key") or f.get("field") or ""
        values = (
            f.get("values")
            or f.get("vals")
            or ([f.get("val")] if f.get("val") is not None else [])
        )
        if values and isinstance(values[0], list):
            values = [item for sub in values for item in sub]
        values = [v for v in values if v is not None and str(v).strip() != ""]

        if not raw_key or not values:
            continue

        canonical_key = resolve_dimension_key(raw_key)
        if not canonical_key:
            print(f"  [warn] Filter field '{raw_key}' not in CSM -- skipped")
            continue

        normalized_filters.append({
            "col_key":      canonical_key,
            "val":          values[0],
            "vals":         values,
            "op":           f.get("operator", "equals"),
            "is_aggregate": False,
        })
        if len(values) > 1:
            print(f"  [normalizer] multi-value filter: {canonical_key} IN {values}")

    data["filters"] = normalized_filters

    # Detect list intent
    q = question.lower()
    is_list_question = any(kw in q for kw in LIST_TRIGGERS)
    metric_is_count  = data.get("metric", "").endswith("_row_count")
    has_no_sort      = not data.get("sort")
    llm_said_list    = data.get("mode") == "list"

    if (is_list_question and metric_is_count and has_no_sort) or llm_said_list:
        data["mode"] = "list"
        print(f"  [normalizer] mode=list detected")

        if not data["dimensions"] and data.get("metric"):
            base_table = (
                csm["metrics"]
                .get(data["metric"], {})
                .get("sources", [""])[0]
            )
            if base_table:
                id_dim   = f"{base_table}_id"
                name_dim = next(
                    (k for k, v in csm["dimensions"].items()
                     if v.get("source") == base_table and v.get("type") == "string"),
                    None,
                )
                auto_dims = []
                if id_dim in csm["dimensions"]:
                    auto_dims.append(id_dim)
                if name_dim:
                    auto_dims.append(name_dim)
                if auto_dims:
                    data["dimensions"] = auto_dims
                    print(f"  [normalizer] auto-injected dims: {auto_dims}")

    if "intent" in intent_output:
        intent_output["intent"] = data
    else:
        intent_output = data

    return intent_output


# ---------------------------------------------------------------------------
# RAG++ resolver
# FIX: strip dimensions whose column is already inside the metric's compute
#      expression so they don't leak into GROUP BY and inflate row counts.
# ---------------------------------------------------------------------------

def _extract_compute_columns(compute: str) -> set[str]:
    """
    Return the bare column references (table.column) found inside a metric's
    compute expression, e.g. 'COUNT(DISTINCT employee_skills.skill_id)' -> 
    {'employee_skills.skill_id', 'skill_id'}.

    We store both the qualified and unqualified form so we can match against
    whatever shape the dim_node exposes.
    """
    import re
    # grab every word.word token inside the expression
    refs = set(re.findall(r'\b(\w+\.\w+)\b', compute))
    # also add the unqualified column names for fallback matching
    for ref in list(refs):
        refs.add(ref.split('.')[1])
    return refs


def rag_plus_plus_resolver(raw_intent):
    data = raw_intent.get('intent', raw_intent)

    m_key = data.get('metric') or 'row_count'
    metric_node = csm['metrics'].get(m_key)
    if not metric_node:
        available_metrics = list(csm['metrics'].keys())
        raise ValueError(f"Metric '{m_key}' not found in CSM. Available: {available_metrics}")

    base_table = metric_node.get('sources')[0]

    # Columns already consumed by the metric's aggregate expression.
    # Dimensions referencing these must NOT appear in GROUP BY / SELECT
    # because grouping by COUNT(DISTINCT x)'s own column gives count=1 per row.
    compute_cols = _extract_compute_columns(metric_node.get('compute', ''))

    required_tables = set()
    dim_nodes       = []
    skipped_dims    = []

    for d_id in data.get('dimensions', []):
        node = csm['dimensions'].get(d_id)
        if not node:
            continue

        col_ref_qualified   = f"{node['source']}.{node['column']}"
        col_ref_unqualified = node['column']

        # Skip this dimension if its column is the one being counted/aggregated
        if col_ref_qualified in compute_cols or col_ref_unqualified in compute_cols:
            skipped_dims.append(d_id)
            print(f"  [resolver] dropped dimension '{d_id}' -- column '{col_ref_qualified}' "
                  f"is already inside metric compute '{metric_node['compute']}'")
            continue

        dim_nodes.append((d_id, node))
        required_tables.add(node['source'])

    if skipped_dims:
        print(f"  [resolver] skipped {len(skipped_dims)} dimension(s) that overlap the metric aggregate")

    valid_filters = []
    for f in data.get('filters', []):
        col_key = f.get('col_key')

        # Accept filter if it has at least one usable value in val OR vals.
        # LLM sometimes sets val=null but populates vals=[...] for multi-value
        # filters, or vice-versa. We treat either as valid.
        val  = f.get('val')
        vals = f.get('vals') or []
        has_value = (
            (val is not None and str(val).strip() != '')
            or any(str(v).strip() != '' for v in vals if v is not None)
        )
        if not has_value:
            continue

        f_dim = csm['dimensions'].get(col_key)
        if not f_dim:
            print(f"  [warn] Filter col_key '{col_key}' not in CSM dimensions -- skipped")
            continue

        required_tables.add(f_dim['source'])
        valid_filters.append(f)

    relationships = csm.get('relationships', {})

    def find_join_path(start, target):
        if start == target:
            return []
        queue   = [(start, [])]
        visited = {start}
        while queue:
            (current_node, path) = queue.pop(0)
            for rel_id, rel in relationships.items():
                if rel['from'] == current_node and rel['to'] not in visited:
                    new_path = path + [(rel['to'], rel['join'])]
                    if rel['to'] == target:
                        return new_path
                    visited.add(rel['to'])
                    queue.append((rel['to'], new_path))
                elif rel['to'] == current_node and rel['from'] not in visited:
                    new_path = path + [(rel['from'], rel['join'])]
                    if rel['from'] == target:
                        return new_path
                    visited.add(rel['from'])
                    queue.append((rel['from'], new_path))
        return None

    active_joins  = []
    joined_tables = {base_table}

    for table in required_tables:
        if table not in joined_tables:
            path = find_join_path(base_table, table)
            if path:
                for to_table, join_on in path:
                    if to_table not in joined_tables:
                        active_joins.append(f"LEFT JOIN {to_table} ON {join_on}")
                        joined_tables.add(to_table)
            else:
                print(f"  [warn] No join path found from {base_table} to {table}")

    dim_col_refs  = [f"{node['source']}.{node['column']}" for _, node in dim_nodes]
    aggregate_sel = dim_col_refs + [f"{metric_node['compute']} AS result"]
    group_by      = dim_col_refs if dim_col_refs else []

    plan = {
        "select":   aggregate_sel,
        "from":     base_table,
        "joins":    active_joins,
        "group_by": group_by,
        "limit":    data.get('limit'),
        "sort":     data.get('sort'),
        "filters":  valid_filters,
    }

    if data.get("mode") == "list":
        plan["mode"] = "list"
        if dim_col_refs:
            plan["select_cols"] = dim_col_refs
        else:
            plan["select_cols"] = [f"{base_table}.*"]

    return plan


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

analytics_pipeline = (
    RunnableParallel({
        "question":        RunnablePassthrough(),
        "metrics_list":    lambda x: list(csm['metrics'].keys()),
        "dimensions_list": lambda x: list(csm['dimensions'].keys()),
        "schema_context":  lambda x: json.dumps({
            "metrics":    {k: {"sources": v['sources']} for k, v in csm['metrics'].items()},
            "dimensions": {k: {"source":  v['source']}  for k, v in csm['dimensions'].items()},
        }),
        "bgo_context":     lambda x: build_bgo_context(glossary),
    })
    | RunnableParallel({
        "intent":   decomposition_chain,
        "question": lambda x: x["question"],
    })
    | RunnableLambda(lambda x: {
        "intent":   enforce_ranking(x['intent'], x['question']),
        "question": x['question'],
    })
    | RunnableLambda(lambda x: {
        "intent":   normalize_intent(x['intent'], x['question']),
        "question": x['question'],
    })
    | RunnableLambda(lambda x: {
        "intent":   enforce_status_filter(x['intent'], x['question']),
        "question": x['question'],
    })
    | RunnableLambda(lambda x: {
        "logical_plan": rag_plus_plus_resolver(x['intent']),
        "intent":       x['intent'],
    })
    | RunnableLambda(lambda x: {
        "sql":          sql_compiler(x['logical_plan']),
        "logical_plan": x['logical_plan'],
        "intent":       x['intent'],
    })
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ask_database(query):
    print(f"Processing: {query}")
    print("Running pipeline...")

    output = analytics_pipeline.invoke(query)

    final_sql    = output['sql']
    logical_plan = output['logical_plan']

    print(f"Generated SQL:\n{final_sql}\n")

    with engine.connect() as conn:
        result = conn.execute(text(final_sql))
        return [dict(row._mapping) for row in result]


def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


if __name__ == "__main__":
    question = input("Ask your question: ")
    results  = ask_database(question)
    print("Data Output:", json.dumps(results, indent=2, default=decimal_default))