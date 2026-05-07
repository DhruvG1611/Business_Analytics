"""
connector.py  --  HR Analytics Pipeline with Cube.js
Schema driven entirely by Cube YAML files -- no external CSM/SQLAlchemy needed.
"""

import yaml
import json
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# --- Cube.js API -------------------------------------------------------------

CUBE_API_URL = "http://localhost:4000/cubejs-api/v1/load"

# --- Load & Parse Cube YAML Schemas ------------------------------------------

CUBE_YAML_FILES = [
    "model/cubes/departments.yml",
    "model/cubes/employees.yml",
    "model/cubes/projects.yml",
    "model/cubes/tasks.yml",
]


def load_cube_schemas(yaml_files: list[str]) -> dict:
    """
    Parse all Cube YAML files into a unified schema registry.

    Registry shape:
        {
            "cubes": {
                "employees": {
                    "sql_table":  "test.employees",
                    "measures":   {"count": {"type": "count"}},
                    "dimensions": {"emp_name": {"type": "string", "primary_key": False}},
                    "joins":      [raw YAML join dicts],
                },
                ...
            },
            "all_measures":   ["employees.count", ...],    # PKs excluded
            "all_dimensions": ["employees.emp_name", ...], # PKs excluded
            "join_map": {
                "tasks": [{"to": "projects", "sql": "...", "relationship": "many_to_one"}]
            },
        }
    """
    registry = {"cubes": {}, "all_measures": [], "all_dimensions": [], "join_map": {}}

    for path in yaml_files:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        for cube in raw.get("cubes", []):
            cube_name = cube["name"]

            measures = {m["name"]: {"type": m["type"]} for m in cube.get("measures", [])}
            dimensions = {
                d["name"]: {
                    "type":        d["type"],
                    "primary_key": d.get("primary_key", False),
                }
                for d in cube.get("dimensions", [])
            }
            raw_joins = cube.get("joins", [])

            registry["cubes"][cube_name] = {
                "sql_table":  cube.get("sql_table", ""),
                "measures":   measures,
                "dimensions": dimensions,
                "joins":      raw_joins,
            }

            for m_name in measures:
                registry["all_measures"].append(f"{cube_name}.{m_name}")

            for d_name, d_meta in dimensions.items():
                if not d_meta["primary_key"]:
                    registry["all_dimensions"].append(f"{cube_name}.{d_name}")

            # Normalise: raw YAML join uses "name", we store as "to"
            if raw_joins:
                registry["join_map"][cube_name] = [
                    {
                        "to":           j["name"],
                        "sql":          j["sql"],
                        "relationship": j["relationship"],
                    }
                    for j in raw_joins
                ]

    return registry


SCHEMA = load_cube_schemas(CUBE_YAML_FILES)

MEASURES_LIST   = SCHEMA["all_measures"]
DIMENSIONS_LIST = SCHEMA["all_dimensions"]


# --- LLM ---------------------------------------------------------------------

llm = ChatOllama(model="llama3", temperature=0, format="json")


# =============================================================================
# 1. DECOMPOSITION CHAIN
# =============================================================================

_DECOMPOSITION_TEMPLATE = """
You are a Precise Semantic Parser for an HR database backed by Cube.js.
Translate the user question into a JSON intent object.

Available measures (use exactly as written):
{measures_list}

Available dimensions (use exactly as written):
{dimensions_list}

### CUBE SCHEMA CONTEXT (auto-generated from Cube YAML):
{schema_context}

### RULES:
1. Return ONLY valid JSON -- no markdown, no explanation.
2. "measure"    : pick ONE from the measures list above.
3. "dimensions" : list of zero or more strings from the dimensions list.
4. "filters"    : only add when a SPECIFIC value is explicitly mentioned.
   Filter object format:
     {{"member": "<cube>.<dimension>", "operator": "equals", "values": ["<value>"]}}
   Supported operators: equals, notEquals, contains, gt, gte, lt, lte
5. "limit"      : integer or null.
6. "sort"       : "asc" | "desc" | null.

### MEASURE SELECTION GUIDE:
- "list / show all <entity>"      --> measure: "<entity>.count", dimensions: ["<entity>.<name_field>"]
- "<entity> per / by department"  --> measure: "<entity>.count", dimensions: ["departments.dept_name"]
- "tasks per employee / workload" --> measure: "tasks.count",    dimensions: ["employees.emp_name"]
- "total count of <entity>"       --> measure: "<entity>.count", dimensions: []
- "most / top / highest"          --> sort: "desc", limit: 1
- "least / lowest / fewest"       --> sort: "asc",  limit: 1

### EXAMPLES:

Q: "list all employees"
A: {{"intent": {{"measure": "employees.count", "dimensions": ["employees.emp_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "show all projects"
A: {{"intent": {{"measure": "projects.count", "dimensions": ["projects.project_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "how many employees are in each department"
A: {{"intent": {{"measure": "employees.count", "dimensions": ["departments.dept_name"], "filters": [], "limit": null, "sort": null}}}}

Q: "which department has the most projects"
A: {{"intent": {{"measure": "projects.count", "dimensions": ["departments.dept_name"], "filters": [], "limit": 1, "sort": "desc"}}}}

Q: "total number of tasks"
A: {{"intent": {{"measure": "tasks.count", "dimensions": [], "filters": [], "limit": null, "sort": null}}}}

Q: "tasks assigned to Alice"
A: {{"intent": {{"measure": "tasks.count", "dimensions": ["tasks.task_name"], "filters": [{{"member": "employees.emp_name", "operator": "equals", "values": ["Alice"]}}], "limit": null, "sort": null}}}}

QUESTION: {question}
RESPONSE:
"""


def _build_schema_context(schema: dict) -> str:
    """Render human-readable schema block for the LLM prompt."""
    lines = []
    for cube_name, meta in schema["cubes"].items():
        dims     = [d for d, m in meta["dimensions"].items() if not m["primary_key"]]
        measures = list(meta["measures"].keys())
        lines.append(
            f"Cube '{cube_name}': measures=[{', '.join(measures)}]  "
            f"dimensions=[{', '.join(dims)}]"
        )
        for j in schema["join_map"].get(cube_name, []):
            lines.append(f"  join -> '{j['to']}' ({j['relationship']}) on {j['sql']}")
    return "\n".join(lines)


decomposition_chain = (
    ChatPromptTemplate.from_template(_DECOMPOSITION_TEMPLATE) | llm | JsonOutputParser()
)


# =============================================================================
# 2. ENFORCE RANKING
# =============================================================================

RANKING_KEYWORDS = {
    "desc": ["most", "highest", "top", "best", "largest", "maximum", "max", "biggest", "busiest"],
    "asc":  ["least", "lowest", "fewest", "smallest", "minimum", "min", "lightest"],
}

NOISE_PHRASES = [
    "by all", "across all", "over all", "overall", "for all",
    "generated by", "in total", "in all", "across the",
]


def enforce_ranking(intent_output: dict, question: str) -> dict:
    data = intent_output.get("intent", intent_output)
    q    = question.lower()

    for direction, keywords in RANKING_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            if not data.get("sort"):
                data["sort"] = direction
                print(f"  [enforcer] sort={direction} (keyword match)")
            if not data.get("limit"):
                data["limit"] = 1
                print(f"  [enforcer] limit=1 (keyword match)")
            break

    if any(phrase in q for phrase in NOISE_PHRASES):
        if data.get("dimensions"):
            print("  [enforcer] dimensions cleared (noise phrase)")
            data["dimensions"] = []

    if "intent" in intent_output:
        intent_output["intent"] = data
    else:
        intent_output = data

    return intent_output


# =============================================================================
# 3. RAG++ RESOLVER
# =============================================================================

def _get_cube_from_member(member: str) -> str:
    return member.split(".")[0]


def _find_join_path(start: str, target: str, join_map: dict) -> list | None:
    """BFS over Cube YAML join edges (treated as bidirectional)."""
    if start == target:
        return []

    adjacency: dict[str, list[dict]] = {}
    for from_cube, edges in join_map.items():
        adjacency.setdefault(from_cube, [])
        for edge in edges:
            adjacency[from_cube].append({"to": edge["to"], "sql": edge["sql"]})
            adjacency.setdefault(edge["to"], [])
            adjacency[edge["to"]].append({"to": from_cube, "sql": edge["sql"]})

    queue   = [(start, [])]
    visited = {start}

    while queue:
        current, path = queue.pop(0)
        for edge in adjacency.get(current, []):
            nxt      = edge["to"]
            new_path = path + [(nxt, edge)]
            if nxt == target:
                return new_path
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, new_path))

    return None


def _best_from_cube(required_cubes: set[str], join_map: dict) -> str:
    """Pick the FROM cube that minimises total joins needed."""
    if len(required_cubes) == 1:
        return next(iter(required_cubes))

    best, best_score = "tasks", float("inf")
    for candidate in required_cubes:
        others  = required_cubes - {candidate}
        missing = sum(1 for t in others if _find_join_path(candidate, t, join_map) is None)
        if missing < best_score:
            best, best_score = candidate, missing

    return best


def rag_plus_plus_resolver(raw_intent: dict, schema: dict) -> dict:
    """Converts LLM intent -> validated Cube.js query plan."""
    data      = raw_intent.get("intent", raw_intent)
    join_map  = schema["join_map"]
    cubes_def = schema["cubes"]

    # Validate measure
    measure = data.get("measure", "")
    if not measure or measure not in schema["all_measures"]:
        raise ValueError(f"Measure '{measure}' not in schema. Available: {schema['all_measures']}")

    measure_cube = _get_cube_from_member(measure)

    # Validate dimensions
    valid_dims = []
    for d in data.get("dimensions", []):
        if d in schema["all_dimensions"]:
            valid_dims.append(d)
        else:
            print(f"  [warn] Dimension '{d}' not in schema -- skipped")

    # Collect required cubes
    required_cubes: set[str] = {measure_cube}
    for d in valid_dims:
        required_cubes.add(_get_cube_from_member(d))
    for f in data.get("filters", []):
        member = f.get("member", "")
        if member:
            cube      = _get_cube_from_member(member)
            dim_field = member.split(".", 1)[1] if "." in member else ""
            if cube in cubes_def and dim_field in cubes_def[cube]["dimensions"]:
                required_cubes.add(cube)
            else:
                print(f"  [warn] Filter member '{member}' not in schema -- skipped")

    # Resolve joins
    from_cube    = _best_from_cube(required_cubes, join_map)
    joined_cubes = {from_cube}
    resolved_joins: list[str] = []

    for target_cube in sorted(required_cubes - {from_cube}):
        path = _find_join_path(from_cube, target_cube, join_map)
        if path is not None:
            for to_cube, edge in path:
                if to_cube not in joined_cubes:
                    resolved_joins.append(edge["sql"])
                    joined_cubes.add(to_cube)
        else:
            print(f"  [warn] No join path from '{from_cube}' to '{target_cube}'")

    # "employees.emp_name" -> "employees.emp_name"
    # Cube name is used exactly as declared in the YAML (no case change).
    def to_cube_member(dotted: str) -> str:
        return dotted  # cube name already matches YAML declaration

    cube_measure    = to_cube_member(measure)
    cube_dimensions = [to_cube_member(d) for d in valid_dims]

    cube_filters = []
    for f in data.get("filters", []):
        member = f.get("member", "")
        values = f.get("values", [])
        op     = f.get("operator", "equals")
        if not member or not values or values == [""]:
            continue
        cube      = _get_cube_from_member(member)
        dim_field = member.split(".", 1)[1] if "." in member else ""
        if cube in cubes_def and (
            dim_field in cubes_def[cube]["dimensions"]
            or dim_field in cubes_def[cube]["measures"]
        ):
            cube_filters.append({
                "member":   to_cube_member(member),
                "operator": op,
                "values":   [str(v) for v in values],
            })
        else:
            print(f"  [warn] Filter '{member}' invalid -- skipped")

    return {
        "measures":   [cube_measure],
        "dimensions": cube_dimensions,
        "filters":    cube_filters,
        "limit":      data.get("limit"),
        "sort":       data.get("sort"),
        "from_cube":  from_cube,
        "joins_used": resolved_joins,
    }


# =============================================================================
# 4. CUBE.JS QUERY BUILDER
# =============================================================================

def cubejs_query_builder(plan: dict) -> dict:
    query: dict = {"measures": plan["measures"]}

    if plan["dimensions"]:
        query["dimensions"] = plan["dimensions"]

    if plan["filters"]:
        query["filters"] = plan["filters"]

    if plan["limit"]:
        query["limit"] = plan["limit"]

    if plan["sort"]:
        query["order"] = {plan["measures"][0]: plan["sort"].upper()}

    return query


# =============================================================================
# 5. CUBE.JS API EXECUTOR
# =============================================================================

def query_cube(cube_query: dict) -> dict | None:
    """
    POST to Cube.js /load.
    Cube.js REST API requires the query wrapped as: {"query": {...}}
    On error, prints the full Cube.js error body for diagnosis.
    """
    payload = {"query": cube_query}

    try:
        response = requests.post(
            CUBE_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            print(f"[Cube.js] HTTP {response.status_code} -- error body:")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            return None

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"[Cube.js] Request failed: {e}")
        return None


# =============================================================================
# 6. PIPELINE
# =============================================================================

analytics_pipeline = (
    RunnableParallel({
        "question":        RunnablePassthrough(),
        "measures_list":   lambda _: "\n".join(f"  - {m}" for m in MEASURES_LIST),
        "dimensions_list": lambda _: "\n".join(f"  - {d}" for d in DIMENSIONS_LIST),
        "schema_context":  lambda _: _build_schema_context(SCHEMA),
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
        "plan":   rag_plus_plus_resolver(x["intent"], SCHEMA),
        "intent": x["intent"],
    })
    | RunnableLambda(lambda x: {
        "cube_query": cubejs_query_builder(x["plan"]),
        "plan":       x["plan"],
        "intent":     x["intent"],
    })
)


# =============================================================================
# 7. PUBLIC INTERFACE
# =============================================================================

def ask_database(question: str) -> list:
    """Execute a natural language question against Cube.js."""
    print(f"\n[?] {question}")

    output     = analytics_pipeline.invoke(question)
    cube_query = output["cube_query"]

    print(f"[Cube.js Query]\n{json.dumps(cube_query, indent=2)}\n")

    result = query_cube(cube_query)

    if result:
        print(f"[Cube.js Response]\n{json.dumps(result, indent=2)}\n")
        return result.get("data", [])

    print("[!] Query returned no data")
    return []


if __name__ == "__main__":
    question = input("Ask your question: ")
    results  = ask_database(question)
    print("[Results]", json.dumps(results, indent=2))