import re
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
# Input sanitiser
# Reject non-natural-language inputs before they hit the LLM pipeline.
# ---------------------------------------------------------------------------

# Patterns that indicate the user typed a shell command or path, not a question.
_SHELL_PATTERNS = [
    re.compile(r'^\s*cd\s+', re.IGNORECASE),
    re.compile(r'^\s*(ls|dir|pwd|mkdir|rm|cp|mv|cat|echo|python|pip|git)\b', re.IGNORECASE),
    re.compile(r'^\s*[a-zA-Z]:\\'),          # Windows absolute path  C:\...
    re.compile(r'^\s*/[a-zA-Z0-9_/]+\s*$'),  # Unix absolute path
]

_MIN_QUESTION_LENGTH = 3


class NotAQuestionError(ValueError):
    """Raised when the user input looks like a shell command, not a question."""


def validate_question(question: str) -> str:
    """
    Raise NotAQuestionError if the input looks like a shell command or is too
    short to be a meaningful question.  Returns the stripped question otherwise.
    """
    q = question.strip()

    if len(q) < _MIN_QUESTION_LENGTH:
        raise NotAQuestionError(
            f"Input '{q}' is too short to be a question. "
            "Please type a natural-language question about your data."
        )

    for pattern in _SHELL_PATTERNS:
        if pattern.match(q):
            raise NotAQuestionError(
                f"Input looks like a shell command: '{q}'. "
                "Please type a natural-language question, e.g. "
                "'Who is the most efficient employee?'"
            )

    return q


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
# FIX: defensive .get() on all intent_pattern fields — a pattern without
#      'metric' (e.g. a list-only pattern) no longer raises KeyError.
# ---------------------------------------------------------------------------

def build_bgo_context(glossary: dict) -> str:
    lines = []

    lines.append("## METRIC SYNONYMS")
    lines.append("If the user's phrasing matches any synonym, use that metric key exactly.\n")
    for metric_key, synonyms in glossary.get("metrics", {}).items():
        if not isinstance(synonyms, list):
            continue
        lines.append(f"  {metric_key}:")
        lines.append(f"    triggers: {', '.join(synonyms)}")
    lines.append("")

    lines.append("## DIMENSION SYNONYMS")
    lines.append("If the user's phrasing matches any synonym, use that dimension key exactly.\n")
    for dim_key, synonyms in glossary.get("dimensions", {}).items():
        if not isinstance(synonyms, list):
            continue
        lines.append(f"  {dim_key}:")
        lines.append(f"    triggers: {', '.join(synonyms)}")
    lines.append("")

    lines.append("## ENTITY DEFINITIONS")
    for entity, meta in glossary.get("ontology", {}).get("entities", {}).items():
        if not isinstance(meta, dict):
            continue
        syns = ", ".join(meta.get("synonyms", []))
        lines.append(f"  {entity}: {meta.get('description', '')}")
        lines.append(f"    also called: {syns}")
    lines.append("")

    lines.append("## ENTITY RELATIONSHIPS")
    lines.append("Use these to determine which dimension to group by.\n")
    for rel in glossary.get("ontology", {}).get("relationships", []):
        if not isinstance(rel, dict):
            continue
        nl = "; ".join(rel.get("natural_language", []))
        lines.append(f"  {rel.get('statement', '')}")
        lines.append(f"    natural language triggers: {nl}")
    lines.append("")

    lines.append("## INTENT PATTERNS")
    lines.append("Concrete mappings from question shape to metric + dimensions.\n")
    for p in glossary.get("intent_patterns", []):
        if not isinstance(p, dict):
            continue
        # FIX: use .get() with defaults — no more KeyError on missing 'metric'
        pattern = p.get('pattern', '')
        metric  = p.get('metric', '')
        dims    = p.get('dimensions', [])
        mode    = p.get('mode')
        filters = p.get('filters')
        sort    = p.get('sort')
        limit   = p.get('limit')

        lines.append(f"  pattern : \"{pattern}\"")
        if metric:
            lines.append(f"  metric  : {metric}")
        if mode:
            lines.append(f"  mode    : {mode}")
        lines.append(f"  dims    : {dims}")
        if filters:
            lines.append(f"  filters : {filters}")
        if sort:
            lines.append(f"  sort    : {sort}, limit: {limit if limit is not None else 1}")
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
                ops_map = {'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'notEquals': '!='}
                clauses.append(f"{col_ref} {ops_map.get(op, '=')} {num_vals[0]}")
            else:
                in_list = ", ".join(str(v) for v in num_vals)
                clauses.append(f"{col_ref} IN ({in_list})")

        elif dtype == 'time':
            safe_val = str(values[0]).replace("'", "''")
            ops_map  = {'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=', 'notEquals': '!='}
            clauses.append(f"{col_ref} {ops_map.get(op, '=')} '{safe_val}'")

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
        cols = plan.get("select_cols") or [f"{plan['from']}.*"]
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
# List-intent and ranking keywords
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
# ---------------------------------------------------------------------------

_STATUS_KEYWORD_MAP = {
    # ── done ──────────────────────────────────────────────────────────────
    "completed":       "done",
    "complete":        "done",
    "done":            "done",
    "finished":        "done",
    "closed":          "done",
    # ── in_progress ───────────────────────────────────────────────────────
    # "working" removed — too generic ("people working on it" is NOT a status filter)
    # "active"  removed — also used in project status queries
    # "started" removed — ambiguous context
    "in progress":     "in_progress",
    "in_progress":     "in_progress",
    "inprogress":      "in_progress",
    "ongoing":         "in_progress",
    "currently doing": "in_progress",
    "being worked on": "in_progress",
    # ── pending ───────────────────────────────────────────────────────────
    "pending":         "pending",
    "open":            "pending",
    "not started":     "pending",
    "waiting":     "pending",
    "blocked":     "blocked",
    "stuck":       "blocked",
}


def enforce_status_filter(intent_output: dict, question: str) -> dict:
    data = intent_output.get('intent', intent_output)
    q    = question.lower()

    matched_statuses: list[str] = []
    for phrase in sorted(_STATUS_KEYWORD_MAP, key=len, reverse=True):
        if phrase in q and _STATUS_KEYWORD_MAP[phrase] not in matched_statuses:
            matched_statuses.append(_STATUS_KEYWORD_MAP[phrase])

    if not matched_statuses:
        if 'intent' in intent_output:
            intent_output['intent'] = data
        else:
            intent_output = data
        return intent_output

    print(f"  [status-enforcer] detected statuses in question: {matched_statuses}")

    existing = [
        f for f in data.get('filters', [])
        if f.get('col_key') != 'tasks_task_status'
    ]

    status_filter = {
        "col_key":      "tasks_task_status",
        "val":          matched_statuses[0],
        "vals":         matched_statuses,
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

    # Normalise filter keys + values (supports multi-value IN filters)
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
    q                = question.lower()
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
# ---------------------------------------------------------------------------

def _extract_compute_columns(compute: str) -> set[str]:
    refs = set(re.findall(r'\b(\w+\.\w+)\b', compute))
    for ref in list(refs):
        refs.add(ref.split('.')[1])
    return refs


def rag_plus_plus_resolver(raw_intent):
    data = raw_intent.get('intent', raw_intent)

    m_key = data.get('metric') or ''

    # ── Metric fallback: never hard-crash on an unknown metric key ──────────
    # The LLM sometimes invents plausible-sounding keys that don't exist
    # (e.g. "total_proficiency_per_project").  We try three recovery strategies
    # before giving up so the pipeline always returns something useful.
    metric_node = csm['metrics'].get(m_key)
    if not metric_node and m_key:
        print(f"  [resolver] metric '{m_key}' not in CSM — attempting fallback")

        # Strategy 1: suffix match — "total_proficiency_per_project" has suffix
        # "per_project"; find any real metric that ends the same way.
        suffix = m_key.split('_per_')[-1] if '_per_' in m_key else ''
        if suffix:
            candidates = [k for k in csm['metrics'] if k.endswith(f'_per_{suffix}')]
            if candidates:
                m_key = candidates[0]
                metric_node = csm['metrics'][m_key]
                print(f"  [resolver] suffix fallback -> '{m_key}'")

        # Strategy 2: token overlap — pick the real metric whose key shares the
        # most words with the invented key.
        if not metric_node:
            invented_tokens = set(m_key.lower().split('_'))
            best_key, best_score = '', 0
            for k in csm['metrics']:
                score = len(invented_tokens & set(k.lower().split('_')))
                if score > best_score:
                    best_score, best_key = score, k
            if best_key and best_score >= 2:
                m_key = best_key
                metric_node = csm['metrics'][m_key]
                print(f"  [resolver] token-overlap fallback ({best_score} tokens) -> '{m_key}'")

        # Strategy 3: hard fallback — use the generic row count for the table
        # most likely implied by the invented key name.
        if not metric_node:
            for tname in csm.get('relationships', {}):
                pass   # just need any table name
            fallback_candidates = [k for k in csm['metrics'] if k.endswith('_row_count')]
            if fallback_candidates:
                m_key = fallback_candidates[0]
                metric_node = csm['metrics'][m_key]
                print(f"  [resolver] last-resort fallback -> '{m_key}'")

    if not metric_node:
        available = list(csm['metrics'].keys())
        raise ValueError(
            f"Metric '{m_key}' not found in CSM and all fallbacks exhausted. "
            f"Available metrics: {available}"
        )

    base_table   = metric_node.get('sources')[0]
    compute_cols = _extract_compute_columns(metric_node.get('compute', ''))

    required_tables = set()
    dim_nodes       = []

    for d_id in data.get('dimensions', []):
        node = csm['dimensions'].get(d_id)
        if not node:
            continue

        col_ref_qualified   = f"{node['source']}.{node['column']}"
        col_ref_unqualified = node['column']

        if col_ref_qualified in compute_cols or col_ref_unqualified in compute_cols:
            print(f"  [resolver] dropped dimension '{d_id}' -- overlaps metric compute")
            continue

        dim_nodes.append((d_id, node))
        required_tables.add(node['source'])

    valid_filters = []
    for f in data.get('filters', []):
        col_key = f.get('col_key')
        val     = f.get('val')
        vals    = f.get('vals') or []
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

    # ── Path hints: metrics can declare an explicit ordered join chain ────────
    # This prevents the BFS from taking a wrong shortcut through an unrelated
    # relationship (e.g. employee_skills -> employees -> departments -> projects
    # instead of employee_skills -> employees -> tasks -> projects).
    #
    # Format in csm_enterprise.yaml under a metric:
    #   join_path: [table_a, table_b, table_c]   # ordered, base_table first
    metric_join_path = metric_node.get('join_path', [])

    def find_join_path(start, target, forbidden_via: set = frozenset()):
        """
        BFS from start to target, optionally avoiding certain intermediate nodes.
        forbidden_via lets us block wrong shortcut paths when a join_path hint
        is active.
        """
        if start == target:
            return []
        queue   = [(start, [])]
        visited = {start}
        while queue:
            current_node, path = queue.pop(0)
            for rel in relationships.values():
                if rel['from'] == current_node and rel['to'] not in visited:
                    next_node = rel['to']
                    if next_node in forbidden_via and next_node != target:
                        continue
                    new_path = path + [(next_node, rel['join'])]
                    if next_node == target:
                        return new_path
                    visited.add(next_node)
                    queue.append((next_node, new_path))
                elif rel['to'] == current_node and rel['from'] not in visited:
                    next_node = rel['from']
                    if next_node in forbidden_via and next_node != target:
                        continue
                    new_path = path + [(next_node, rel['join'])]
                    if next_node == target:
                        return new_path
                    visited.add(next_node)
                    queue.append((next_node, new_path))
        return None

    active_joins  = []
    joined_tables = {base_table}

    if metric_join_path:
        # Follow the explicit path in order, regardless of what BFS would find
        print(f"  [resolver] using explicit join_path hint: {metric_join_path}")
        chain = metric_join_path if metric_join_path[0] == base_table else [base_table] + metric_join_path
        for i in range(len(chain) - 1):
            frm, to = chain[i], chain[i + 1]
            if to in joined_tables:
                continue
            # Find the direct relationship between these two adjacent tables
            join_clause = None
            for rel in relationships.values():
                if (rel['from'] == frm and rel['to'] == to) or                    (rel['to'] == frm and rel['from'] == to):
                    join_clause = rel['join']
                    break
            if join_clause:
                active_joins.append(f"LEFT JOIN {to} ON {join_clause}")
                joined_tables.add(to)
            else:
                print(f"  [warn] join_path hint: no direct relationship between '{frm}' and '{to}'")
        # Any required_tables not covered by the path hint get normal BFS,
        # but we forbid the wrong intermediate nodes the hint was avoiding.
        hinted_tables = set(chain)
        all_tables_in_model = set(
            t for rel in relationships.values() for t in (rel['from'], rel['to'])
        )
        # Tables NOT in the hint path are potential wrong shortcuts to block
        forbidden = all_tables_in_model - hinted_tables - required_tables
        for table in required_tables:
            if table not in joined_tables:
                path = find_join_path(base_table, table, forbidden_via=forbidden)
                if path:
                    for to_table, join_on in path:
                        if to_table not in joined_tables:
                            active_joins.append(f"LEFT JOIN {to_table} ON {join_on}")
                            joined_tables.add(to_table)
                else:
                    print(f"  [warn] No join path found from {base_table} to {table}")
    else:
        # Standard BFS — no hints
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
        plan["select_cols"] = dim_col_refs if dim_col_refs else [f"{base_table}.*"]

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

def ask_database(query: str):
    # Guard: reject shell commands and non-questions before touching the LLM
    query = validate_question(query)

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
    while True:
        try:
            question = input("Ask your question (or 'exit' to quit): ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            results = ask_database(question)
            print("Data Output:", json.dumps(results, indent=2, default=decimal_default))
        except NotAQuestionError as e:
            print(f"\n[input error] {e}\n")
        except KeyboardInterrupt:
            break