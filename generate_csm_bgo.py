"""
generate_csm_bgo.py
-------------------
Inspects a live database via SQLAlchemy, then:
  1. Builds csm_enterprise.yaml  (deterministic, from schema introspection)
  2. Generates bgo.yaml           (via Ollama)
  3. Renders decomposition_prompt.txt  (from CSM + BGO, ready for connector.py)

Usage:
    python generate_csm_bgo.py

Configure DB_URL below (or set env var DATABASE_URL) then run once.
Re-run whenever your schema changes — all three files are regenerated together.

connector.py loads decomposition_prompt.txt at startup:
    with open("decomposition_prompt.txt") as f:
        _PROMPT_TEMPLATE = f.read()
    decomposition_chain = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE) | llm | JsonOutputParser()

FIXES vs original:
  - PKs are now included as dimensions (dtype="id") so keys like
    employees_id are in the CSM and LLM guesses resolve correctly.
  - build_decomposition_prompt emits a "list" mode example so the LLM
    learns to return mode="list" for "show / list all" questions.
"""

import os
import re
import sys
import yaml
from itertools import combinations

from sqlalchemy import create_engine, inspect, text
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===========================================================================
# CONFIG
# ===========================================================================

DB_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost:3306/test",
)

OLLAMA_MODEL = "llama3"

EXCLUDE_TABLES      = {"alembic_version", "django_migrations", "flyway_schema_history"}
FK_HEURISTIC_SUFFIX = ("_id",)


# ===========================================================================
# STEP 1 -- Introspect the database
# ===========================================================================

def introspect_db(engine) -> dict:
    insp   = inspect(engine)
    tables = {}

    for tname in insp.get_table_names():
        if tname in EXCLUDE_TABLES:
            continue

        cols    = insp.get_columns(tname)
        pk_cols = set(insp.get_pk_constraint(tname).get("constrained_columns", []))
        fk_map  = {}

        for fk in insp.get_foreign_keys(tname):
            for local_col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fk_map[local_col] = f"{fk['referred_table']}.{ref_col}"

        for col in cols:
            cname = col["name"]
            if cname not in fk_map and cname not in pk_cols:
                for suffix in FK_HEURISTIC_SUFFIX:
                    if cname.endswith(suffix) and cname != suffix:
                        guessed_table = cname[: -len(suffix)] + "s"
                        fk_map[cname] = f"{guessed_table}.id"
                        break

        column_meta = []
        for col in cols:
            cname = col["name"]
            column_meta.append({
                "name":     cname,
                "type":     str(col["type"]),
                "pk":       cname in pk_cols,
                "nullable": col.get("nullable", True),
                "fk":       fk_map.get(cname),
            })

        row_count     = 0
        sample_values = {}
        try:
            with engine.connect() as conn:
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM `{tname}`")).scalar()
                for col in column_meta:
                    if col["pk"] or col["fk"]:
                        continue
                    ctype = col["type"].upper()
                    if any(t in ctype for t in ("CHAR", "TEXT", "ENUM")):
                        rows = conn.execute(
                            text(f"SELECT DISTINCT `{col['name']}` FROM `{tname}` LIMIT 5")
                        ).fetchall()
                        vals = [r[0] for r in rows if r[0] is not None]
                        if vals:
                            sample_values[col["name"]] = vals
        except Exception as e:
            print(f"  [warn] Could not sample {tname}: {e}")

        tables[tname] = {
            "columns":      column_meta,
            "foreign_keys": [
                {"col": c, "ref_table": v.split(".")[0], "ref_col": v.split(".")[1]}
                for c, v in fk_map.items()
            ],
            "row_count":     row_count,
            "sample_values": sample_values,
        }

    return {"tables": tables}


# ===========================================================================
# STEP 2 -- Build CSM deterministically
# ===========================================================================

def build_csm(schema: dict) -> dict:
    metrics       = {}
    dimensions    = {}
    relationships = {}

    for tname, tmeta in schema["tables"].items():
        metrics[f"{tname}_row_count"] = {
            "compute": "COUNT(*)",
            "sources": [tname],
            "label":   f"Total {tname.replace('_', ' ').title()}",
        }

        for col in tmeta["columns"]:
            cname     = col["name"]
            ctype_raw = col["type"].upper()

            # FIX: PKs are no longer skipped — they get dtype "id" so the LLM
            # can reference them in "list" queries (e.g. employees_id).
            if col["pk"]:
                dtype = "id"
            elif any(t in ctype_raw for t in ("INT", "BIGINT", "SMALLINT", "TINYINT",
                                               "DECIMAL", "FLOAT", "DOUBLE", "NUMERIC")):
                dtype = "number"
            elif any(t in ctype_raw for t in ("DATE", "TIME", "DATETIME", "TIMESTAMP")):
                dtype = "time"
            else:
                dtype = "string"

            dim_entry = {
                "source": tname,
                "column": cname,
                "type":   dtype,
                "label":  cname.replace("_", " ").title(),
            }
            if dtype == "string" and cname in tmeta.get("sample_values", {}):
                dim_entry["sample_values"] = tmeta["sample_values"][cname]
            if col.get("fk"):
                dim_entry["fk"] = col["fk"]

            dimensions[f"{tname}_{cname}"] = dim_entry

        for fk in tmeta["foreign_keys"]:
            rel_key = f"{tname}_to_{fk['ref_table']}"
            relationships[rel_key] = {
                "from": tname,
                "to":   fk["ref_table"],
                "join": f"{tname}.{fk['col']} = {fk['ref_table']}.{fk['ref_col']}",
            }

    table_names = list(schema["tables"].keys())
    for t1, t2 in combinations(table_names, 2):
        rel_fwd = f"{t1}_to_{t2}"
        rel_rev = f"{t2}_to_{t1}"
        if rel_fwd in relationships or rel_rev in relationships:
            rel    = relationships.get(rel_fwd) or relationships.get(rel_rev)
            child  = rel["from"]
            parent = rel["to"]
            key    = f"{child}_per_{parent.rstrip('s')}"
            metrics[key] = {
                "compute": f"COUNT(DISTINCT {child}.id)",
                "sources": [child, parent],
                "label":   f"{child.title()} per {parent.rstrip('s').title()}",
            }

    return {"metrics": metrics, "dimensions": dimensions, "relationships": relationships}


# ===========================================================================
# STEP 3 -- Generate BGO via Ollama
# ===========================================================================

def _schema_summary(schema: dict) -> str:
    lines = []
    for tname, tmeta in schema["tables"].items():
        non_pk_cols = [c for c in tmeta["columns"] if not c["pk"]]
        col_desc    = ", ".join(
            f"{c['name']} ({c['type']}{'->'+c['fk'] if c.get('fk') else ''})"
            for c in non_pk_cols
        )
        lines.append(f"Table '{tname}' ({tmeta['row_count']} rows): {col_desc}")
        for col, vals in tmeta.get("sample_values", {}).items():
            lines.append(f"  sample {col}: {vals}")
    return "\n".join(lines)


def _csm_summary(csm: dict) -> str:
    lines = ["Metrics: " + ", ".join(csm["metrics"].keys())]
    lines.append("Dimensions: " + ", ".join(csm["dimensions"].keys()))
    lines.append("Relationships:")
    for rv in csm["relationships"].values():
        lines.append(f"  {rv['from']} -> {rv['to']}  on {rv['join']}")
    return "\n".join(lines)


BGO_PROMPT = """
You are an expert data analyst and ontologist.

Given the database schema and CSM below, generate a Business Glossary & Ontology
YAML file. Follow this EXACT structure — no deviations:

metrics:
  <csm_metric_key>:
    - "<natural language synonym 1>"
    - "<natural language synonym 2>"

dimensions:
  <csm_dimension_key>:
    - "<natural language synonym 1>"
    - "<natural language synonym 2>"

ontology:
  entities:
    <table_name>:
      description: "<one sentence in business terms>"
      synonyms:
        - "<synonym>"

  relationships:
    - statement: "<plain English, e.g. An employee belongs to one department.>"
      from: <from_table>
      to: <to_table>
      cardinality: many_to_one
      natural_language:
        - "<phrase implying this join>"

intent_patterns:
  - pattern: "<question template>"
    metric: "<csm_metric_key>"
    dimensions:
      - "<csm_dimension_key>"

DATABASE SCHEMA:
{schema_summary}

GENERATED CSM:
{csm_summary}

RULES:
- Output ONLY valid YAML. No markdown fences, no explanation, no preamble.
- Every metric and dimension key must exactly match the CSM keys above.
- 4-8 synonyms per metric/dimension.
- 6-10 intent_patterns covering: list all, total count, per-dimension grouping, top/bottom 1, filter by value.
- intent_patterns must use only keys that exist in the CSM.
- For "list all" patterns, add mode: list to the intent_pattern entry.
"""


def generate_bgo_with_ollama(schema: dict, csm: dict) -> dict:
    llm   = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    chain = ChatPromptTemplate.from_template(BGO_PROMPT) | llm | StrOutputParser()

    print("  Calling Ollama to generate BGO (this may take 30-60s)...")
    raw = chain.invoke({
        "schema_summary": _schema_summary(schema),
        "csm_summary":    _csm_summary(csm),
    })

    raw = re.sub(r"^```(?:yaml)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw.strip(), flags=re.MULTILINE)

    try:
        bgo = yaml.safe_load(raw)
        if not isinstance(bgo, dict):
            raise ValueError("Parsed BGO is not a dict")
        return bgo
    except Exception as e:
        print(f"  [warn] BGO YAML parse failed ({e}) -- saving raw output")
        with open("bgo_raw_ollama_output.txt", "w") as f:
            f.write(raw)
        return {}


# ===========================================================================
# STEP 4 -- Validate CSM
# ===========================================================================

def validate_csm(csm: dict, schema: dict) -> list[str]:
    warnings   = []
    all_tables = set(schema["tables"].keys())

    for m_key, m_val in csm["metrics"].items():
        for src in m_val.get("sources", []):
            if src not in all_tables:
                warnings.append(f"Metric '{m_key}' references unknown table '{src}'")

    for d_key, d_val in csm["dimensions"].items():
        if d_val["source"] not in all_tables:
            warnings.append(f"Dimension '{d_key}' references unknown table '{d_val['source']}'")
        else:
            col_names = {c["name"] for c in schema["tables"][d_val["source"]]["columns"]}
            if d_val["column"] not in col_names:
                warnings.append(
                    f"Dimension '{d_key}' references unknown column "
                    f"'{d_val['source']}.{d_val['column']}'"
                )

    for r_key, r_val in csm["relationships"].items():
        if r_val["from"] not in all_tables:
            warnings.append(f"Relationship '{r_key}': from-table '{r_val['from']}' not found")
        if r_val["to"] not in all_tables:
            warnings.append(f"Relationship '{r_key}': to-table '{r_val['to']}' not found")

    return warnings


# ===========================================================================
# STEP 5 -- Build decomposition prompt from CSM + BGO
# ===========================================================================

def _build_selection_guide(csm: dict, glossary: dict) -> str:
    lines        = ["SELECTION GUIDE:"]
    bgo_dim_syns = glossary.get("dimensions", {})

    parent_dim: dict[str, str] = {}
    for rel in csm.get("relationships", {}).values():
        to_table = rel.get("to", "")
        if not to_table or to_table in parent_dim:
            continue
        for dim_key, dim_val in csm.get("dimensions", {}).items():
            if dim_val.get("source") == to_table and dim_val.get("type") == "string":
                parent_dim[to_table] = dim_key
                break

    for table, dim_key in parent_dim.items():
        synonyms   = bgo_dim_syns.get(dim_key, [table.rstrip("s")])
        short_syns = synonyms[:3]
        phrase     = " / ".join(short_syns)
        lines.append(f'- "per / by {phrase}"  ->  dimensions: ["{dim_key}"]')

    lines.append('- "most / top / highest / max"       ->  sort: "desc", limit: 1')
    lines.append('- "least / lowest / fewest / min"    ->  sort: "asc",  limit: 1')
    lines.append('- "list / show / display / all / get all / every"  ->  mode: "list", dimensions: [<id_col>, <name_col>]')
    lines.append('- No grouping requested (total count)              ->  dimensions: []')
    return "\n".join(lines)


def _format_intent(metric: str, dims: list, filters: list, sort, limit,
                   mode: str | None = None) -> str:
    dims_str = str(dims).replace("'", '"')
    if not filters:
        filters_str = "[]"
    else:
        parts = []
        for f in filters:
            field  = f.get("field", "")
            op     = f.get("operator", "equals")
            values = f.get("values", ["<value>"])
            val    = values[0] if values else "<value>"
            parts.append(f'{{"field": "{field}", "operator": "{op}", "values": ["{val}"]}}')
        filters_str = "[" + ", ".join(parts) + "]"

    limit_str = "null" if limit is None else str(limit)
    sort_str  = "null" if sort  is None else f'"{sort}"'
    mode_part = f', "mode": "{mode}"' if mode else ''
    return (
        f'{{"metric": "{metric}", "dimensions": {dims_str}, '
        f'"filters": {filters_str}, "limit": {limit_str}, "sort": {sort_str}{mode_part}}}'
    )


def _build_examples(csm: dict, glossary: dict) -> str:
    lines    = ["EXAMPLES:"]
    metrics  = csm.get("metrics", {})
    dims     = csm.get("dimensions", {})
    patterns = glossary.get("intent_patterns", [])
    m_syns   = glossary.get("metrics", {})

    if patterns:
        for p in patterns:
            m_key     = p.get("metric", "")
            p_dims    = [d for d in p.get("dimensions", []) if d in dims]
            p_filters = p.get("filters", [])
            p_sort    = p.get("sort")
            p_limit   = p.get("limit", 1) if p_sort else None
            p_mode    = p.get("mode")

            if m_key not in metrics:
                continue

            entity = m_key.split("_")[0]
            q      = p.get("pattern", f"list all {entity}").replace("{entity}", entity)

            lines.append(f'\nQ: "{q}"')
            lines.append(
                f'A: {{"intent": {_format_intent(m_key, p_dims, p_filters, p_sort, p_limit, p_mode)}}}'
            )
    else:
        # Minimal fallback — one list + one count example per table
        for m_key, m_val in list(metrics.items())[:8]:
            if "_per_" in m_key:
                continue
            source   = m_val.get("sources", [""])[0]
            id_dim   = f"{source}_id"
            name_dim = next(
                (k for k, v in dims.items()
                 if v.get("source") == source and v.get("type") == "string"),
                None
            )
            q_syns = m_syns.get(m_key, [])
            q      = q_syns[0] if q_syns else f"list all {source}"

            list_dims = []
            if id_dim in dims:
                list_dims.append(id_dim)
            if name_dim:
                list_dims.append(name_dim)

            lines.append(f'\nQ: "{q}"')
            lines.append(
                f'A: {{"intent": {_format_intent(m_key, list_dims, [], None, None, "list")}}}'
            )

            # Also add a count example
            lines.append(f'\nQ: "how many {source} are there?"')
            lines.append(
                f'A: {{"intent": {_format_intent(m_key, [], [], None, None, None)}}}'
            )

    # FIX: Always append a hard-coded list example so the LLM knows the pattern
    lines.append('\n# List mode examples (always emit mode="list" for show/list/display/get questions):')
    for tname in list({v["source"] for v in csm.get("dimensions", {}).values()})[:3]:
        id_dim   = f"{tname}_id"
        name_dim = next(
            (k for k, v in dims.items()
             if v.get("source") == tname and v.get("type") == "string"),
            None
        )
        count_metric = f"{tname}_row_count"
        if count_metric not in metrics:
            continue
        list_dims = []
        if id_dim in dims:
            list_dims.append(id_dim)
        if name_dim:
            list_dims.append(name_dim)
        lines.append(f'\nQ: "list all {tname}"')
        lines.append(
            f'A: {{"intent": {_format_intent(count_metric, list_dims, [], None, None, "list")}}}'
        )
        lines.append(f'\nQ: "show me all {tname} with their id"')
        lines.append(
            f'A: {{"intent": {_format_intent(count_metric, list_dims, [], None, None, "list")}}}'
        )

    return "\n".join(lines)


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def build_decomposition_prompt(csm: dict, glossary: dict) -> str:
    selection_guide = _escape_braces(_build_selection_guide(csm, glossary))
    examples        = _escape_braces(_build_examples(csm, glossary))
    filter_example  = '{{"field": "<dimension_key>", "operator": "equals", "values": ["<value>"]}}'

    parts = [
        "You are a precise semantic parser for a business analytics system.",
        "Translate the user's question into a JSON intent object using ONLY the provided",
        "metrics and dimensions. Resolve synonyms using the Business Glossary below.",
        "",
        "Available metrics (use exactly as written):",
        "{metrics_list}",
        "",
        "Available dimensions (use exactly as written):",
        "{dimensions_list}",
        "",
        "DATA MODEL CONTEXT (from CSM YAML):",
        "{schema_context}",
        "",
        "BUSINESS GLOSSARY & ONTOLOGY (read live from bgo.yaml):",
        "{bgo_context}",
        "",
        "RULES:",
        "1. Return ONLY valid JSON -- no markdown, no explanation.",
        "2. \"metric\"     : pick ONE key from the metrics list. Use METRIC SYNONYMS above",
        "                  to resolve informal phrasing.",
        "3. \"dimensions\" : zero or more keys from the dimensions list. Use DIMENSION",
        "                  SYNONYMS to resolve informal phrasing. Use ENTITY RELATIONSHIPS",
        "                  to decide which dimension to group by.",
        "4. \"filters\"    : ONLY add when a SPECIFIC value is explicitly mentioned.",
        f"   Filter format: {filter_example}",
        "   Supported operators: equals, notEquals, contains, gt, gte, lt, lte",
        "5. \"limit\"      : integer or null.",
        "6. \"sort\"       : \"asc\" | \"desc\" | null.",
        "7. \"mode\"       : \"list\" when the question asks to show/list/display/get rows.",
        "                  Omit (or null) for aggregate/count questions.",
        "                  When mode=list, include the id dimension (e.g. employees_id)",
        "                  and any name/label dimension in the dimensions array.",
        "",
        selection_guide,
        "",
        examples,
        "",
        "QUESTION: {question}",
        "RESPONSE:",
    ]

    return "\n".join(parts)


# ===========================================================================
# HELPERS
# ===========================================================================

def _dump_yaml(data: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, indent=2)
    print(f"  Written: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("CSM + BGO + Prompt Generator")
    print("=" * 60)

    # 1. Connect
    print(f"\n[1/5] Connecting to: {DB_URL.split('@')[-1]}")
    try:
        engine = create_engine(DB_URL, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Connected OK")
    except Exception as e:
        print(f"  ERROR: Cannot connect -- {e}")
        sys.exit(1)

    # 2. Introspect
    print("\n[2/5] Introspecting schema...")
    schema = introspect_db(engine)
    tables = schema["tables"]
    print(f"  Found {len(tables)} tables: {', '.join(tables.keys())}")
    for tname, tmeta in tables.items():
        col_names = [c["name"] for c in tmeta["columns"]]
        fks       = [f"{f['col']} -> {f['ref_table']}.{f['ref_col']}" for f in tmeta["foreign_keys"]]
        print(f"    {tname}: cols={col_names}" + (f"  FKs={fks}" if fks else ""))

    # 3. Build + validate CSM
    print("\n[3/5] Building CSM...")
    csm = build_csm(schema)
    print(f"  Metrics: {len(csm['metrics'])}  "
          f"Dimensions: {len(csm['dimensions'])}  "
          f"Relationships: {len(csm['relationships'])}")

    warnings = validate_csm(csm, schema)
    if warnings:
        print("  VALIDATION WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  Validation: OK")

    _dump_yaml(csm, "csm_enterprise.yaml")

    # 4. Generate BGO
    print("\n[4/5] Generating BGO via Ollama...")
    bgo = generate_bgo_with_ollama(schema, csm)

    if bgo:
        bgo.setdefault("metrics", {})
        bgo.setdefault("dimensions", {})
        for mk in csm["metrics"]:
            bgo["metrics"].setdefault(mk, [mk.replace("_", " ")])
        for dk in csm["dimensions"]:
            bgo["dimensions"].setdefault(dk, [dk.replace("_", " ")])
        _dump_yaml(bgo, "bgo.yaml")
    else:
        print("  BGO generation failed -- check bgo_raw_ollama_output.txt")
        print("  Generating prompt with stub BGO...")
        bgo = {
            "metrics":    {mk: [mk.replace("_", " ")] for mk in csm["metrics"]},
            "dimensions": {dk: [dk.replace("_", " ")] for dk in csm["dimensions"]},
            "ontology":   {"entities": {}, "relationships": []},
            "intent_patterns": [],
        }

    # 5. Build decomposition prompt
    print("\n[5/5] Building decomposition prompt...")
    prompt = build_decomposition_prompt(csm, bgo)

    with open("decomposition_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("  Written: decomposition_prompt.txt")

    print("\nDone. Files written:")
    print("  csm_enterprise.yaml")
    print("  bgo.yaml")
    print("  decomposition_prompt.txt")


if __name__ == "__main__":
    main()