import yaml
import json
import decimal
from guardrail import validate_sql
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
        lines.append("")

    return "\n".join(lines)


engine = create_engine(
    "mysql+pymysql://root:@localhost:3306/test",
    future=True,
)

llm = ChatOllama(model="llama3", temperature=0, format="json")

decomposition_chain = ChatPromptTemplate.from_template("""
You are a precise semantic parser for an HR analytics system.
Translate the user's question into a JSON intent object using ONLY the provided
metrics and dimensions. Resolve synonyms using the Business Glossary below.

Available metrics (use exactly as written):
{metrics_list}

Available dimensions (use exactly as written):
{dimensions_list}

DATA MODEL CONTEXT (from CSM YAML):
{schema_context}

BUSINESS GLOSSARY & ONTOLOGY:
{bgo_context}

RULES:
1. Return ONLY valid JSON -- no markdown, no explanation.
2. "metric"     : pick ONE key from the metrics list. Use the synonym table above
                  to resolve informal phrasing (e.g. "headcount" -> employees_row_count).
3. "dimensions" : zero or more keys from the dimensions list. Use the synonym table
                  to resolve informal phrasing (e.g. "division" -> Departments_dept_name).
                  Use the RELATIONSHIPS section to decide which dimension to group by
                  (e.g. "per department" -> Departments_dept_name).
4. "filters"    : ONLY add when a SPECIFIC value is explicitly mentioned.
   Filter format: {{"field": "<dimension_key>", "operator": "equals", "values": ["<value>"]}}
   Supported operators: equals, notEquals, contains, gt, gte, lt, lte
5. "limit"      : integer or null.
6. "sort"       : "asc" | "desc" | null.

SELECTION GUIDE (synonyms resolved via BGO):
- Any phrasing that maps to a metric synonym         -> use that metric key
- "per / by department / division / unit"            -> dimensions: ["Departments_dept_name"]
- "per / by employee / person / staff / resource"    -> dimensions: ["Employees_emp_name"]
- "per / by project / initiative / program"          -> dimensions: ["Projects_project_name"]
- "most / top / highest / busiest / largest / max"   -> sort: "desc", limit: 1
- "least / lowest / fewest / smallest / min"         -> sort: "asc",  limit: 1
- No grouping requested (total count only)           -> dimensions: []

EXAMPLES (derived from BGO intent patterns):

Q: "list all employees"
A: {{"intent": {{"metric": "employees_row_count", "dimensions": ["Employees_emp_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "show all projects"
A: {{"intent": {{"metric": "projects_row_count", "dimensions": ["Projects_project_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "headcount by division"
A: {{"intent": {{"metric": "employees_row_count", "dimensions": ["Departments_dept_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "how many staff are in each unit"
A: {{"intent": {{"metric": "employees_row_count", "dimensions": ["Departments_dept_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "which department has the most projects"
A: {{"intent": {{"metric": "projects_row_count", "dimensions": ["Departments_dept_name"], "filters": [], "limit": 1, "sort": "desc"}}}}

Q: "who has the most tasks"
A: {{"intent": {{"metric": "tasks_row_count", "dimensions": ["Employees_emp_name"], "filters": [], "limit": 1, "sort": "desc"}}}}

Q: "total number of tasks"
A: {{"intent": {{"metric": "tasks_row_count", "dimensions": [], "filters": [], "limit": null, "sort": null}}}}

Q: "assignments"
A: {{"intent": {{"metric": "tasks_row_count", "dimensions": [], "filters": [], "limit": null, "sort": null}}}}

Q: "tasks assigned to Alice"
A: {{"intent": {{"metric": "tasks_row_count", "dimensions": ["Projects_project_name"], "filters": [{{"field": "Employees_emp_name", "operator": "equals", "values": ["Alice"]}}], "limit": null, "sort": null}}}}

Q: "employees in the Engineering department"
A: {{"intent": {{"metric": "employees_row_count", "dimensions": ["Employees_emp_name"], "filters": [{{"field": "Departments_dept_name", "operator": "equals", "values": ["Engineering"]}}], "limit": null, "sort": null}}}}

Q: "backlog"
A: {{"intent": {{"metric": "tasks_row_count", "dimensions": [], "filters": [], "limit": null, "sort": null}}}}

Q: "workforce size per team"
A: {{"intent": {{"metric": "employees_row_count", "dimensions": ["Departments_dept_name"], "filters": [], "limit": null, "sort": null}}}}

QUESTION: {question}
RESPONSE:
""") | llm | JsonOutputParser()


def sql_compiler(plan):
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
    if where_filters:
        clauses = []
        for f in where_filters:
            dim_node = csm['dimensions'].get(f['col_key'])
            if dim_node:
                col = dim_node['column']
                if dim_node.get('is_time'):
                    col_ref = col
                else:
                    col_ref = f"{dim_node['source']}.{col}"
                val = str(f['val']).replace("'", "''")
                clauses.append(f"LOWER({col_ref}) LIKE LOWER('{val}')")
            else:
                print(f"  [warn] Filter key '{f['col_key']}' not in CSM dimensions")
        if clauses:
            sql += f"\nWHERE {' AND '.join(clauses)}"

    if plan.get('group_by'):
        sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    having_filters = [f for f in plan.get('filters', []) if f.get('is_aggregate')]
    if having_filters:
        clauses = []
        for f in having_filters:
            metric = csm['metrics'].get(f['col_key'])
            if metric:
                clauses.append(f"{metric['compute']} {f['op']} {f['val']}")
        if clauses:
            sql += f"\nHAVING {' AND '.join(clauses)}"

    if plan.get('sort') == 'desc':
        sql += "\nORDER BY result DESC"
    elif plan.get('sort') == 'asc':
        sql += "\nORDER BY result ASC"

    if plan.get('limit'):
        sql += f"\nLIMIT {plan['limit']}"

    return sql


RANKING_KEYWORDS = {
    "desc": ["most", "highest", "top", "best", "costliest", "expensive", "largest", "maximum", "max", "biggest"],
    "asc":  ["least", "lowest", "cheapest", "smallest", "minimum", "min", "fewest"],
}


def enforce_ranking(intent_output: dict, question: str) -> dict:
    data = intent_output.get('intent', intent_output)
    q = question.lower()

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


def rag_plus_plus_resolver(raw_intent):
    data = raw_intent.get('intent', raw_intent)

    m_key = data.get('metric') or 'row_count'
    metric_node = csm['metrics'].get(m_key)
    if not metric_node:
        available_metrics = list(csm['metrics'].keys())
        raise ValueError(f"Metric '{m_key}' not found in CSM. Available: {available_metrics}")

    base_table = metric_node.get('sources')[0]

    required_tables = set()
    dim_nodes = []
    for d_id in data.get('dimensions', []):
        node = csm['dimensions'].get(d_id)
        if node:
            dim_nodes.append(node)
            required_tables.add(node['source'])

    valid_filters = []
    for f in data.get('filters', []):
        col_key = f.get('col_key')
        val = f.get('val')

        if not val or (isinstance(val, str) and val.strip() == ''):
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
        queue = [(start, [])]
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

    active_joins = []
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

    return {
        "select":   [f"{d['source']}.{d['column']}" for d in dim_nodes] + [f"{metric_node['compute']} AS result"],
        "from":     base_table,
        "joins":    active_joins,
        "group_by": [f"{d['source']}.{d['column']}" for d in dim_nodes],
        "limit":    data.get('limit'),
        "sort":     data.get('sort'),
        "filters":  data.get('filters', []),
    }


def normalize_intent(intent_output):
    data = intent_output.get('intent', intent_output)

    # Build case-insensitive lookup for dimensions
    dim_lookup = {k.lower(): k for k in csm['dimensions'].keys()}

    # --- Normalize dimensions ---
    normalized_dims = []
    for d in data.get("dimensions", []):
        if not d:
            continue

        canonical = dim_lookup.get(d.lower())
        if canonical:
            normalized_dims.append(canonical)
        else:
            print(f"  [warn] Dimension '{d}' not in CSM (case-insensitive check failed)")

    data["dimensions"] = normalized_dims

    # --- Normalize filters ---
    normalized_filters = []
    for f in data.get("filters", []):
        raw_key = f.get("col_key") or f.get("field")
        values  = f.get("values") or ([f.get("val")] if f.get("val") else [])

        if not raw_key or not values:
            continue

        canonical_key = dim_lookup.get(raw_key.lower())

        if not canonical_key:
            print(f"  [warn] Filter key '{raw_key}' not in CSM dimensions (case-insensitive) -- skipped")
            continue

        normalized_filters.append({
            "col_key":      canonical_key,
            "val":          values[0],
            "op":           f.get("operator", "equals"),
            "is_aggregate": False,
        })

    data["filters"] = normalized_filters

    if "intent" in intent_output:
        intent_output["intent"] = data
    else:
        intent_output = data

    return intent_output


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
        "intent":   normalize_intent(x['intent']),
        "question": x['question'],
    })
    | RunnableLambda(lambda x: {
        "logical_plan": rag_plus_plus_resolver(x['intent']),
        "intent":       x['intent'],
    })
    | RunnableLambda(lambda x: {
        "sql":          sql_compiler(x['logical_plan']),
        "logical_plan": x['logical_plan'],   # carried through for guardrail
        "intent":       x['intent'],
    })
)


def ask_database(query):
    print(f"Processing: {query}")
    print("Running pipeline...")

    output = analytics_pipeline.invoke(query)

    final_sql    = output['sql']
    logical_plan = output['logical_plan']   # now in scope, no NameError

    print(f"Generated SQL:\n{final_sql}\n")

    # Guardrail
   
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