"""
generate_csm_bgo.py
-------------------
Inspects a live database via SQLAlchemy, then uses Ollama (llama3) to
automatically write two YAML files that plug directly into your connector:

  csm_enterprise.yaml  -- Canonical Semantic Model (Module 2 of BI Architecture)
  bgo.yaml             -- Business Glossary & Ontology  (Module 3)

Usage:
    python generate_csm_bgo.py

Configure DB_URL below (or set env var DATABASE_URL) then run once.
Re-run whenever your schema changes to keep the CSM in sync.
"""

import os
import re
import sys
import json
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
    "mysql+pymysql://root:@localhost:3306/test",   # <-- change if needed
)

OLLAMA_MODEL = "llama3"

# Tables to skip entirely (system / audit tables)
EXCLUDE_TABLES = {"alembic_version", "django_migrations", "flyway_schema_history"}

# Column name patterns that are almost certainly foreign keys even if
# SQLAlchemy doesn't see an explicit FK constraint (common in MySQL MyISAM)
FK_HEURISTIC_SUFFIX = ("_id",)

# ===========================================================================
# STEP 1 -- Introspect the database
# ===========================================================================

def introspect_db(engine) -> dict:
    """
    Returns a rich schema dict:
    {
      "tables": {
        "employees": {
          "columns": [
            {"name": "id",      "type": "INTEGER", "pk": True,  "nullable": False},
            {"name": "emp_name","type": "VARCHAR", "pk": False, "nullable": False},
            {"name": "dept_id", "type": "INTEGER", "pk": False, "nullable": True,
             "fk": "departments.id"},
          ],
          "foreign_keys": [
            {"col": "dept_id", "ref_table": "departments", "ref_col": "id"}
          ],
          "row_count": 42,
          "sample_values": {"emp_name": ["Alice", "Bob"], "dept_id": [1, 2]}
        },
        ...
      }
    }
    """
    insp   = inspect(engine)
    tables = {}

    for tname in insp.get_table_names():
        if tname in EXCLUDE_TABLES:
            continue

        cols     = insp.get_columns(tname)
        pk_cols  = set(insp.get_pk_constraint(tname).get("constrained_columns", []))
        fk_map   = {}   # col_name -> "ref_table.ref_col"

        for fk in insp.get_foreign_keys(tname):
            for local_col, ref_col in zip(
                fk["constrained_columns"], fk["referred_columns"]
            ):
                fk_map[local_col] = f"{fk['referred_table']}.{ref_col}"

        # Heuristic FK detection for engines without FK metadata (MyISAM etc.)
        for col in cols:
            cname = col["name"]
            if cname not in fk_map and cname not in pk_cols:
                for suffix in FK_HEURISTIC_SUFFIX:
                    if cname.endswith(suffix) and cname != suffix:
                        # guess: employee.dept_id  -> departments.id
                        guessed_table = cname[: -len(suffix)] + "s"   # naive pluralisation
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

        # Row count + sample values (non-PK, non-FK string/text cols only)
        row_count     = 0
        sample_values = {}
        try:
            with engine.connect() as conn:
                row_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {tname}")
                ).scalar()

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
            "row_count":    row_count,
            "sample_values": sample_values,
        }

    return {"tables": tables}


# ===========================================================================
# STEP 2 -- Build CSM deterministically from introspection
# ===========================================================================

def build_csm(schema: dict) -> dict:
    """
    Constructs csm_enterprise.yaml content as a Python dict.

    Structure:
      metrics:
        <table>_row_count:
          compute: COUNT(*)
          sources: [<table>]
          label: ...
      dimensions:
        <table>_<col>:
          source: <table>
          column: <col>
          type: string|number|time
          label: ...
      relationships:
        <from>_to_<to>:
          from: <table>
          to: <ref_table>
          join: "<table>.<col> = <ref_table>.<ref_col>"
    """
    metrics       = {}
    dimensions    = {}
    relationships = {}

    for tname, tmeta in schema["tables"].items():
        # -- Metrics ----------------------------------------------------------
        # Basic count for every table
        metrics[f"{tname}_row_count"] = {
            "compute": "COUNT(*)",
            "sources": [tname],
            "label":   f"Total {tname.replace('_', ' ').title()}",
        }

        # -- Dimensions -------------------------------------------------------
        for col in tmeta["columns"]:
            cname = col["name"]
            if col["pk"]:
                continue   # PKs are structural, not analytical dimensions

            ctype_raw = col["type"].upper()
            if any(t in ctype_raw for t in ("INT", "BIGINT", "SMALLINT", "TINYINT", "DECIMAL", "FLOAT", "DOUBLE", "NUMERIC")):
                dtype = "number"
            elif any(t in ctype_raw for t in ("DATE", "TIME", "DATETIME", "TIMESTAMP")):
                dtype = "time"
            else:
                dtype = "string"

            dim_key = f"{tname}_{cname}"
            dim_entry = {
                "source": tname,
                "column": cname,
                "type":   dtype,
                "label":  cname.replace("_", " ").title(),
            }

            # Attach sample values for string dims — helps LLM filter generation
            if dtype == "string" and cname in tmeta.get("sample_values", {}):
                dim_entry["sample_values"] = tmeta["sample_values"][cname]

            # Mark FK dims so the CSM knows they're join keys
            if col.get("fk"):
                dim_entry["fk"] = col["fk"]

            dimensions[dim_key] = dim_entry

        # -- Relationships ----------------------------------------------------
        for fk in tmeta["foreign_keys"]:
            rel_key = f"{tname}_to_{fk['ref_table']}"
            relationships[rel_key] = {
                "from": tname,
                "to":   fk["ref_table"],
                "join": f"{tname}.{fk['col']} = {fk['ref_table']}.{fk['ref_col']}",
            }

    # Cross-table count metrics (e.g. employees_per_dept)
    table_names = list(schema["tables"].keys())
    for t1, t2 in combinations(table_names, 2):
        rel_fwd = f"{t1}_to_{t2}"
        rel_rev = f"{t2}_to_{t1}"
        if rel_fwd in relationships or rel_rev in relationships:
            rel = relationships.get(rel_fwd) or relationships.get(rel_rev)
            child  = rel["from"]
            parent = rel["to"]
            key    = f"{child}_per_{parent.rstrip('s')}"
            metrics[key] = {
                "compute": f"COUNT(DISTINCT {child}.id)",
                "sources": [child, parent],
                "label":   f"{child.title()} per {parent.rstrip('s').title()}",
            }

    return {
        "metrics":       metrics,
        "dimensions":    dimensions,
        "relationships": relationships,
    }


# ===========================================================================
# STEP 3 -- Use Ollama to enrich + generate BGO
# ===========================================================================

def _schema_summary(schema: dict) -> str:
    """Compact text description of the schema for the LLM prompt."""
    lines = []
    for tname, tmeta in schema["tables"].items():
        non_pk_cols = [c for c in tmeta["columns"] if not c["pk"]]
        col_desc    = ", ".join(
            f"{c['name']} ({c['type']}{'->'+c['fk'] if c.get('fk') else ''})"
            for c in non_pk_cols
        )
        lines.append(f"Table '{tname}' ({tmeta['row_count']} rows): {col_desc}")
        if tmeta.get("sample_values"):
            for col, vals in tmeta["sample_values"].items():
                lines.append(f"  sample {col}: {vals}")
    return "\n".join(lines)


def _csm_summary(csm: dict) -> str:
    """Compact text description of generated CSM for the LLM prompt."""
    lines = ["Metrics: " + ", ".join(csm["metrics"].keys())]
    lines.append("Dimensions: " + ", ".join(csm["dimensions"].keys()))
    lines.append("Relationships:")
    for rk, rv in csm["relationships"].items():
        lines.append(f"  {rv['from']}.{rv['join'].split('=')[0].split('.')[1].strip()} -> {rv['to']}")
    return "\n".join(lines)


BGO_PROMPT = """
You are an expert data analyst and ontologist.

Given the database schema and CSM below, generate a Business Glossary & Ontology
YAML file. Follow this EXACT structure — no deviations:

```yaml
metrics:
  <csm_metric_key>:
    - "<natural language synonym 1>"
    - "<natural language synonym 2>"
    # 4-8 synonyms per metric that a business user might say

dimensions:
  <csm_dimension_key>:
    - "<natural language synonym 1>"
    - "<natural language synonym 2>"
    # 4-8 synonyms per dimension

ontology:
  entities:
    <table_name>:
      description: "<one sentence describing what this entity represents in business terms>"
      synonyms:
        - "<synonym 1>"
        - "<synonym 2>"

  relationships:
    - statement: "<plain English relationship statement, e.g. 'An employee belongs to one department.'"
      from: <from_table>
      to: <to_table>
      cardinality: many_to_one
      natural_language:
        - "<question or phrase that implies this join>"
        - "<another phrase>"

intent_patterns:
  - pattern: "<common question template with {{entity}} placeholders>"
    metric: "<csm_metric_key>"
    dimensions:
      - "<csm_dimension_key>"
  # Include 6-10 patterns covering list, count, per-dimension, top/bottom, filter queries
```

DATABASE SCHEMA:
{schema_summary}

GENERATED CSM:
{csm_summary}

RULES:
- Output ONLY the YAML block. No explanation, no markdown fences, no preamble.
- Every metric key and dimension key must exactly match the CSM keys above.
- Synonyms must reflect real business language for an HR/operations domain.
- intent_patterns must use only metric and dimension keys that exist in the CSM.
"""


def generate_bgo_with_ollama(schema: dict, csm: dict) -> dict:
    """Call Ollama to produce the BGO YAML, parse and return as dict."""
    llm   = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    chain = ChatPromptTemplate.from_template(BGO_PROMPT) | llm | StrOutputParser()

    print("  Calling Ollama to generate BGO (this may take 30-60s)...")
    raw = chain.invoke({
        "schema_summary": _schema_summary(schema),
        "csm_summary":    _csm_summary(csm),
    })

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:yaml)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw.strip(), flags=re.MULTILINE)

    try:
        bgo = yaml.safe_load(raw)
        if not isinstance(bgo, dict):
            raise ValueError("Parsed BGO is not a dict")
        return bgo
    except Exception as e:
        print(f"  [warn] BGO YAML parse failed ({e}) — saving raw output for inspection")
        with open("bgo_raw_ollama_output.txt", "w") as f:
            f.write(raw)
        return {}


# ===========================================================================
# STEP 4 -- Validate CSM completeness
# ===========================================================================

def validate_csm(csm: dict, schema: dict) -> list[str]:
    """
    Run sanity checks on the generated CSM.
    Returns list of warning strings (empty = all good).
    """
    warnings = []
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
# STEP 5 -- Write YAML files
# ===========================================================================

def _dump_yaml(data: dict, path: str):
    """Write dict to YAML with clean formatting."""
    with open(path, "w") as f:
        yaml.dump(
            data, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )
    print(f"  Written: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("CSM + BGO Auto-Generator")
    print("=" * 60)

    # 1. Connect
    print(f"\n[1/4] Connecting to database: {DB_URL.split('@')[-1]}")
    try:
        engine = create_engine(DB_URL, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Connected OK")
    except Exception as e:
        print(f"  ERROR: Cannot connect — {e}")
        sys.exit(1)

    # 2. Introspect
    print("\n[2/4] Introspecting schema...")
    schema = introspect_db(engine)
    tables = schema["tables"]
    print(f"  Found {len(tables)} tables: {', '.join(tables.keys())}")
    for tname, tmeta in tables.items():
        col_names = [c["name"] for c in tmeta["columns"]]
        fks       = [f"{f['col']} -> {f['ref_table']}.{f['ref_col']}" for f in tmeta["foreign_keys"]]
        print(f"    {tname}: cols={col_names}" + (f"  FKs={fks}" if fks else ""))

    # 3. Build CSM
    print("\n[3/4] Building CSM...")
    csm = build_csm(schema)
    print(f"  Metrics:       {len(csm['metrics'])}")
    print(f"  Dimensions:    {len(csm['dimensions'])}")
    print(f"  Relationships: {len(csm['relationships'])}")

    warnings = validate_csm(csm, schema)
    if warnings:
        print("  VALIDATION WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  Validation: OK")

    _dump_yaml(csm, "csm_enterprise.yaml")

    # 4. Generate BGO
    print("\n[4/4] Generating Business Glossary & Ontology via Ollama...")
    bgo = generate_bgo_with_ollama(schema, csm)

    if bgo:
        # Ensure every CSM metric and dimension has at least a stub entry
        bgo.setdefault("metrics", {})
        bgo.setdefault("dimensions", {})
        for mk in csm["metrics"]:
            bgo["metrics"].setdefault(mk, [f"{mk.replace('_', ' ')}"])
        for dk in csm["dimensions"]:
            bgo["dimensions"].setdefault(dk, [dk.replace("_", " ")])

        _dump_yaml(bgo, "bgo.yaml")
    else:
        print("  BGO generation failed — check bgo_raw_ollama_output.txt")

    print("\nDone. Files written:")
    print("  csm_enterprise.yaml")
    print("  bgo.yaml")
    print("\nDrop these into your project folder and restart connector.py.")


if __name__ == "__main__":
    main()