"""
generate_csm_bgo.py
-------------------
Introspects a live MySQL database and auto-generates:
  1. csm_enterprise.yaml  (metrics, dimensions, relationships)
  2. bgo.yaml             (glossary, ontology, intent patterns, views, PII)

Usage:  python generate_csm_bgo.py
"""

import os, re, json, shutil, hashlib
from dataclasses import dataclass, field
from datetime import datetime

import mysql.connector
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from langchain_core.messages import SystemMessage, HumanMessage
import config

# ── Constants ────────────────────────────────────────────────────────────────

DB_CONFIG = dict(host="localhost", user="root", password="", database="sakila", port=3306)
LLM_CACHE_FILE = "llm_cache.json"
AMOUNT_KEYWORDS = {"amount", "price", "cost", "rate", "salary", "revenue", "fee", "total"}
PII_PATTERNS = {"email", "phone", "password", "first_name", "last_name", "address", "picture", "photo"}
PII_SENSITIVITY = {"password": "high", "email": "medium", "phone": "medium", "picture": "medium", "photo": "medium"}

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    table: str
    name: str
    data_type: str
    is_nullable: bool
    is_pk: bool = False
    is_fk: bool = False
    fk_ref_table: str = ""
    fk_ref_column: str = ""
    column_key: str = ""
    extra: str = ""
    comment: str = ""
    default: str = ""
    sample_values: list = field(default_factory=list)

@dataclass
class SchemaInfo:
    tables: dict = field(default_factory=dict)       # table -> [ColumnInfo]
    primary_keys: dict = field(default_factory=dict)  # table -> [col_name]
    foreign_keys: list = field(default_factory=list)  # [{table, column, ref_table, ref_column}]
    views: list = field(default_factory=list)
    procedures: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    triggers: list = field(default_factory=list)       # [{name, event, table}]

# ── LLM cache ────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if os.path.exists(LLM_CACHE_FILE):
        with open(LLM_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def _cache_key(table: str, cols: list, fks: list) -> str:
    blob = json.dumps({"t": table, "c": cols, "f": fks}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

# ── Database introspection ───────────────────────────────────────────────────

def _query(conn, sql):
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows

def introspect_schema() -> SchemaInfo:
    conn = mysql.connector.connect(**DB_CONFIG)
    schema = SchemaInfo()

    # Columns
    rows = _query(conn, """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
               COLUMN_KEY, EXTRA, COLUMN_COMMENT, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME NOT IN (SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA = DATABASE())
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    for r in rows:
        tbl = r["TABLE_NAME"]
        schema.tables.setdefault(tbl, [])
        schema.tables[tbl].append(ColumnInfo(
            table=tbl, name=r["COLUMN_NAME"], data_type=r["DATA_TYPE"],
            is_nullable=(r["IS_NULLABLE"] == "YES"), column_key=r.get("COLUMN_KEY", ""),
            extra=r.get("EXTRA", ""), comment=r.get("COLUMN_COMMENT", ""),
            default=str(r.get("COLUMN_DEFAULT", "") or ""),
        ))

    # Primary keys
    pks = _query(conn, """
        SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() AND CONSTRAINT_NAME = 'PRIMARY'
    """)
    for r in pks:
        schema.primary_keys.setdefault(r["TABLE_NAME"], []).append(r["COLUMN_NAME"])
        for c in schema.tables.get(r["TABLE_NAME"], []):
            if c.name == r["COLUMN_NAME"]:
                c.is_pk = True

    # Foreign keys
    fks = _query(conn, """
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
    """)
    for r in fks:
        schema.foreign_keys.append(r)
        for c in schema.tables.get(r["TABLE_NAME"], []):
            if c.name == r["COLUMN_NAME"]:
                c.is_fk = True
                c.fk_ref_table = r["REFERENCED_TABLE_NAME"]
                c.fk_ref_column = r["REFERENCED_COLUMN_NAME"]

    # Views, procedures, functions, triggers
    schema.views = [r["TABLE_NAME"] for r in _query(conn,
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA = DATABASE()")]
    schema.procedures = [r["ROUTINE_NAME"] for r in _query(conn,
        "SELECT ROUTINE_NAME FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_SCHEMA = DATABASE() AND ROUTINE_TYPE='PROCEDURE'")]
    schema.functions = [r["ROUTINE_NAME"] for r in _query(conn,
        "SELECT ROUTINE_NAME FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_SCHEMA = DATABASE() AND ROUTINE_TYPE='FUNCTION'")]
    schema.triggers = [{"name": r["TRIGGER_NAME"], "event": r["EVENT_MANIPULATION"], "table": r["EVENT_OBJECT_TABLE"]}
        for r in _query(conn,
        "SELECT TRIGGER_NAME, EVENT_MANIPULATION, EVENT_OBJECT_TABLE FROM INFORMATION_SCHEMA.TRIGGERS WHERE TRIGGER_SCHEMA = DATABASE()")]

    conn.close()
    return schema

def fetch_sample_values(schema: SchemaInfo) -> dict:
    conn = mysql.connector.connect(**DB_CONFIG)
    samples = {}
    string_types = {"varchar", "char", "text", "enum", "set", "tinytext", "mediumtext", "longtext"}
    for tbl, cols in schema.tables.items():
        for c in cols:
            if c.data_type.lower() in string_types and not c.is_pk:
                try:
                    rows = _query(conn, f"SELECT DISTINCT `{c.name}` FROM `{tbl}` WHERE `{c.name}` IS NOT NULL LIMIT 5")
                    vals = [str(r[c.name]) for r in rows if r[c.name] is not None]
                    if vals:
                        samples[f"{tbl}.{c.name}"] = vals
                        c.sample_values = vals
                except Exception:
                    pass
    conn.close()
    return samples

# ── LLM calls ────────────────────────────────────────────────────────────────

def call_llm_for_table(table: str, cols: list[ColumnInfo], fk_list: list, cache: dict) -> dict:
    col_strs = [f"{c.name} ({c.data_type}, {'PK' if c.is_pk else 'FK→'+c.fk_ref_table if c.is_fk else 'regular'})" for c in cols]
    fk_strs = [f"{f['COLUMN_NAME']} → {f['REFERENCED_TABLE_NAME']}.{f['REFERENCED_COLUMN_NAME']}" for f in fk_list]
    key = _cache_key(table, [c.name for c in cols], fk_strs)
    if key in cache:
        return cache[key]

    prompt = f"""Given a database table "{table}" with columns: {', '.join(col_strs)}
and foreign keys: {', '.join(fk_strs) if fk_strs else 'none'}
Generate JSON with these keys:
- "entity_description": 2-3 sentence business description
- "entity_synonyms": list of 5 synonyms
- "business_rules": list of 4 business rules as strings
- "column_descriptions": dict of {{column_name: "1-2 sentence description"}}
- "column_synonyms": dict of {{column_name: ["synonym1", "synonym2", "synonym3"]}}
- "relationship_questions": dict of {{fk_column: ["question1", "question2", "question3"]}}
- "intent_patterns": list of 3 objects with keys "pattern" (natural language question), "metric_key" (snake_case), "dimensions" (list of dimension keys as table_column)
Respond ONLY with valid JSON, no markdown, no explanation."""

    try:
        resp = config.llm.invoke([
            SystemMessage(content="You are a data architect. Respond only with valid JSON."),
            HumanMessage(content=prompt),
        ])
        text = resp.content.strip()
        if text.startswith("```"): text = re.sub(r"^```\w*\n?", "", text)
        if text.endswith("```"): text = text.rsplit("```", 1)[0]
        data = json.loads(text.strip())
        cache[key] = data
        _save_cache(cache)
        return data
    except Exception as e:
        print(f"  ⚠ LLM failed for {table}: {e}")
        return {"entity_description": f"Table {table}.", "entity_synonyms": [table],
                "business_rules": [], "column_descriptions": {}, "column_synonyms": {},
                "relationship_questions": {}, "intent_patterns": []}

# ── Type mapping ─────────────────────────────────────────────────────────────

def _sql_to_csm_type(col: ColumnInfo) -> str:
    if col.is_pk or col.is_fk: return "id"
    dt = col.data_type.lower()
    if dt in ("tinyint", "boolean", "bit") and "active" in col.name.lower(): return "boolean"
    if dt in ("int", "smallint", "tinyint", "mediumint", "bigint", "decimal", "float", "double", "numeric", "year"):
        return "number"
    if dt in ("date", "datetime", "timestamp", "time"): return "time"
    return "string"

# ── CSM builder ──────────────────────────────────────────────────────────────

def build_csm(schema: SchemaInfo, llm_data: dict) -> dict:
    metrics = {}
    dimensions = {}
    relationships = {}

    # --- Row count metrics ---
    for tbl in sorted(schema.tables):
        label_parts = llm_data.get(tbl, {})
        metrics[f"{tbl}_row_count"] = {
            "compute": "COUNT(*)", "sources": [tbl],
            "label": f"Total {tbl.replace('_', ' ').title()}s"
        }

    # --- Numeric aggregate metrics ---
    for tbl, cols in schema.tables.items():
        for c in cols:
            if c.is_pk or c.is_fk: continue
            name_lower = c.name.lower()
            if any(kw in name_lower for kw in AMOUNT_KEYWORDS):
                if c.data_type.lower() in ("decimal", "float", "double", "int", "smallint", "mediumint", "bigint", "numeric"):
                    metrics[f"total_{tbl}_{c.name}"] = {
                        "compute": f"SUM({tbl}.{c.name})", "sources": [tbl],
                        "label": f"Total {c.name.replace('_', ' ').title()}"
                    }
                    metrics[f"avg_{tbl}_{c.name}"] = {
                        "compute": f"AVG({tbl}.{c.name})", "sources": [tbl],
                        "label": f"Average {c.name.replace('_', ' ').title()}"
                    }

    # --- Junction table metrics ---
    for tbl, cols in schema.tables.items():
        fk_cols = [c for c in cols if c.is_fk]
        non_pk_non_fk = [c for c in cols if not c.is_pk and not c.is_fk and c.name != "last_update"]
        if len(fk_cols) == 2 and len(non_pk_non_fk) == 0:
            e1, e2 = fk_cols[0], fk_cols[1]
            key1 = f"{e1.fk_ref_table}_per_{e2.fk_ref_table}"
            metrics[key1] = {
                "compute": f"COUNT(DISTINCT {tbl}.{e1.name})", "sources": [tbl, e2.fk_ref_table],
                "label": f"{e1.fk_ref_table.title()}s per {e2.fk_ref_table.title()}",
                "join_path": [tbl, e2.fk_ref_table]
            }
            key2 = f"{e2.fk_ref_table}_per_{e1.fk_ref_table}"
            metrics[key2] = {
                "compute": f"COUNT(DISTINCT {tbl}.{e2.name})", "sources": [tbl, e1.fk_ref_table],
                "label": f"{e2.fk_ref_table.title()}s per {e1.fk_ref_table.title()}",
                "join_path": [tbl, e1.fk_ref_table]
            }

    # --- FK chain revenue metrics ---
    fk_map = {}
    for fk in schema.foreign_keys:
        fk_map.setdefault(fk["TABLE_NAME"], []).append(fk)
    # Find payment chains
    if "payment" in schema.tables:
        def _walk(tbl, path, depth):
            if depth > 5: return []
            paths = []
            for fk in fk_map.get(tbl, []):
                ref = fk["REFERENCED_TABLE_NAME"]
                if ref not in path:
                    new_path = path + [ref]
                    paths.append(new_path)
                    paths.extend(_walk(ref, new_path, depth + 1))
            return paths
        chains = _walk("payment", ["payment"], 0)
        for chain in chains:
            terminal = chain[-1]
            t_cols = schema.tables.get(terminal, [])
            str_cols = [c for c in t_cols if _sql_to_csm_type(c) == "string" and c.name != "last_update"]
            if str_cols and len(chain) >= 3:
                mkey = f"revenue_by_{terminal}"
                if mkey not in metrics:
                    metrics[mkey] = {
                        "compute": "SUM(payment.amount)", "sources": list(chain),
                        "label": f"Revenue by {terminal.replace('_', ' ').title()}",
                        "join_path": list(chain)
                    }

    # --- Dimensions ---
    for tbl in sorted(schema.tables):
        for c in sorted(schema.tables[tbl], key=lambda x: x.name):
            dim_key = f"{tbl}_{c.name}"
            entry = {
                "source": tbl, "column": c.name,
                "type": _sql_to_csm_type(c),
                "label": c.name.replace("_", " ").title(),
            }
            if c.is_nullable: entry["nullable"] = True
            if c.sample_values: entry["sample_values"] = c.sample_values
            if c.is_fk: entry["fk"] = f"{c.fk_ref_table}.{c.fk_ref_column}"
            dimensions[dim_key] = entry

    # --- Relationships ---
    seen = set()
    for fk in schema.foreign_keys:
        key = f"{fk['TABLE_NAME']}_to_{fk['REFERENCED_TABLE_NAME']}"
        if key in seen:
            key += f"_via_{fk['COLUMN_NAME']}"
        seen.add(key)
        relationships[key] = {
            "from": fk["TABLE_NAME"], "to": fk["REFERENCED_TABLE_NAME"],
            "join": f"{fk['TABLE_NAME']}.{fk['COLUMN_NAME']} = {fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}"
        }

    return {"metrics": metrics, "dimensions": dimensions, "relationships": relationships}

# ── BGO builder ──────────────────────────────────────────────────────────────

def build_bgo(schema: SchemaInfo, llm_data: dict, csm: dict) -> dict:
    bgo = {}

    # --- Metadata ---
    db_name = DB_CONFIG["database"]
    bgo["metadata"] = {
        "database": db_name, "domain": f"{db_name.title()} Database",
        "description": f"Auto-generated glossary for the {db_name} database.",
        "version": "1.0",
        "tables": sorted(schema.tables.keys()),
        "views": sorted(schema.views),
        "stored_procedures": sorted(schema.procedures),
        "stored_functions": sorted(schema.functions),
    }

    # --- Ontology ---
    entities = {}
    for tbl in sorted(schema.tables):
        ld = llm_data.get(tbl, {})
        pk_cols = schema.primary_keys.get(tbl, [])
        pk = pk_cols if len(pk_cols) > 1 else (pk_cols[0] if pk_cols else "id")
        entities[tbl] = {
            "description": ld.get("entity_description", f"Table {tbl}."),
            "synonyms": ld.get("entity_synonyms", [tbl.title()]),
            "primary_key": pk,
            "business_rules": ld.get("business_rules", []),
        }

    ont_rels = []
    fk_by_table = {}
    for fk in schema.foreign_keys:
        fk_by_table.setdefault(fk["TABLE_NAME"], []).append(fk)
    for tbl, fks in sorted(fk_by_table.items()):
        refs = list({f["REFERENCED_TABLE_NAME"] for f in fks})
        ld = llm_data.get(tbl, {})
        rq = ld.get("relationship_questions", {})
        nls = []
        for fk in fks:
            nls.extend(rq.get(fk["COLUMN_NAME"], []))
        card = "many_to_many" if len(fks) == 2 and all(c.is_pk for c in schema.tables.get(tbl, []) if c.is_fk) else "many_to_one"
        entry = {
            "statement": f"A {tbl} references {', '.join(refs)}.",
            "from": tbl,
            "to": refs if len(refs) > 1 else refs[0],
            "cardinality": card,
        }
        if nls: entry["natural_language"] = nls[:3]
        ont_rels.append(entry)

    bgo["ontology"] = {"entities": entities, "relationships": ont_rels}

    # --- Dimensions ---
    dims = {}
    for tbl in sorted(schema.tables):
        ld = llm_data.get(tbl, {})
        col_descs = ld.get("column_descriptions", {})
        col_syns = ld.get("column_synonyms", {})
        for c in sorted(schema.tables[tbl], key=lambda x: x.name):
            dim_key = f"{tbl}_{c.name}"
            entry = {
                "label": c.name.replace("_", " ").title(),
                "synonyms": col_syns.get(c.name, [c.name.replace("_", " ").title()]),
                "description": col_descs.get(c.name, f"Column {c.name} in table {tbl}."),
                "data_type": c.data_type.upper(),
                "nullable": c.is_nullable,
            }
            # Domain values for boolean-like
            if "active" in c.name.lower() and c.data_type.lower() in ("tinyint", "boolean", "bit"):
                entry["domain_values"] = {1: "Active", 0: "Inactive"}
            # PII flag
            if c.name.lower() in ("password",):
                entry["pii"] = True
            dims[dim_key] = entry
    bgo["dimensions"] = dims

    # --- Metrics ---
    bgo_metrics = {}
    for mkey, mval in csm["metrics"].items():
        ld_table = mval["sources"][0] if mval.get("sources") else ""
        ld = llm_data.get(ld_table, {})
        bgo_metrics[mkey] = {
            "label": mval.get("label", mkey),
            "synonyms": [mval.get("label", mkey), mkey.replace("_", " ")],
            "description": mval.get("label", mkey),
            "calculation": mval.get("compute", ""),
        }
    bgo["metrics"] = bgo_metrics

    # --- Intent patterns ---
    patterns = []
    for tbl in sorted(schema.tables):
        ld = llm_data.get(tbl, {})
        for p in ld.get("intent_patterns", []):
            if isinstance(p, dict) and "pattern" in p:
                patterns.append({
                    "pattern": p["pattern"],
                    "metric": p.get("metric_key", f"{tbl}_row_count"),
                    "dimensions": p.get("dimensions", []),
                })
    bgo["intent_patterns"] = patterns

    # --- Views ---
    views_dict = {}
    for v in sorted(schema.views):
        views_dict[v] = {
            "description": f"Database view: {v.replace('_', ' ')}.",
            "use_cases": [f"Query {v.replace('_', ' ')} data"],
        }
    bgo["views"] = views_dict

    # --- Stored procedures / functions ---
    sp_dict = {}
    for p in sorted(schema.procedures):
        sp_dict[p] = {"description": f"Stored procedure: {p.replace('_', ' ')}."}
    bgo["stored_procedures"] = sp_dict

    sf_dict = {}
    for f in sorted(schema.functions):
        sf_dict[f] = {"description": f"Stored function: {f.replace('_', ' ')}."}
    bgo["stored_functions"] = sf_dict

    # --- Triggers ---
    trig_dict = {}
    for t in schema.triggers:
        trig_dict[t["name"]] = {
            "table": t["table"], "event": t["event"],
            "description": f"Fires on {t['event']} for table {t['table']}.",
        }
    bgo["triggers"] = trig_dict

    # --- PII fields ---
    pii_list = []
    for tbl, cols in schema.tables.items():
        for c in cols:
            if c.name.lower() in PII_PATTERNS:
                sens = PII_SENSITIVITY.get(c.name.lower(), "low")
                pii_list.append({"table": tbl, "column": c.name, "sensitivity": sens})
    bgo["pii_fields"] = pii_list

    return bgo

# ── YAML writer ──────────────────────────────────────────────────────────────

def write_yaml(data: dict, path: str, header_comment: str = ""):
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, "w", encoding="utf-8") as f:
        if header_comment:
            f.write(header_comment + "\n\n")
        yaml.dump(data, f)

def backup_existing_files():
    for fname in ("csm_enterprise.yaml", "bgo.yaml"):
        if os.path.exists(fname):
            shutil.copy2(fname, fname + ".bak")
            print(f"  ✓ Backed up {fname} → {fname}.bak")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CSM & BGO Generator")
    print("=" * 60)

    # Step 1: Backup
    backup_existing_files()

    # Step 2: Introspect
    print("\n[1/5] Introspecting database schema...")
    schema = introspect_schema()
    total_cols = sum(len(v) for v in schema.tables.values())
    print(f"  ✓ Found {len(schema.tables)} tables, {total_cols} columns, {len(schema.foreign_keys)} foreign keys")
    print(f"  ✓ Found {len(schema.views)} views, {len(schema.procedures)} procedures, {len(schema.functions)} functions, {len(schema.triggers)} triggers")

    # Step 3: Sample values
    print("\n[2/5] Fetching sample values...")
    samples = fetch_sample_values(schema)
    print(f"  ✓ Sampled {len(samples)} string columns")

    # Step 4: LLM enrichment
    print("\n[3/5] LLM enrichment per table...")
    cache = _load_cache()
    cache_before = len(cache)
    llm_data = {}
    for tbl in sorted(schema.tables):
        cols = schema.tables[tbl]
        fk_list = [f for f in schema.foreign_keys if f["TABLE_NAME"] == tbl]
        data = call_llm_for_table(tbl, cols, fk_list, cache)
        llm_data[tbl] = data
        print(f"  ✓ {tbl}")
    cache_hits = cache_before
    cache_new = len(cache) - cache_before
    print(f"  ✓ LLM processed {len(schema.tables)} tables ({cache_hits} from cache, {cache_new} new)")

    # Step 5: Build CSM
    print("\n[4/5] Building CSM...")
    csm_data = build_csm(schema, llm_data)
    print(f"  ✓ Generated {len(csm_data['metrics'])} metrics, {len(csm_data['dimensions'])} dimensions, {len(csm_data['relationships'])} relationships")

    # Step 6: Build BGO
    print("\n[5/5] Building BGO...")
    bgo_data = build_bgo(schema, llm_data, csm_data)
    print(f"  ✓ Generated {len(bgo_data.get('intent_patterns', []))} intent patterns")

    # Step 7: Write YAML
    csm_header = "# ============================================================\n# CSM Enterprise — Auto-Generated\n# Database: {}\n# Generated: {}\n# ============================================================".format(
        DB_CONFIG["database"], datetime.now().isoformat()[:19])
    bgo_header = "# ============================================================\n# Business Glossary & Ontology — Auto-Generated\n# Database: {}\n# Generated: {}\n# ============================================================".format(
        DB_CONFIG["database"], datetime.now().isoformat()[:19])

    write_yaml(csm_data, "csm_enterprise.yaml", csm_header)
    write_yaml(bgo_data, "bgo.yaml", bgo_header)

    print("\n" + "=" * 60)
    print(f"✓ Introspected {len(schema.tables)} tables, {total_cols} columns, {len(schema.foreign_keys)} foreign keys")
    print(f"✓ LLM processed {len(schema.tables)} tables ({cache_hits} from cache, {cache_new} new)")
    print(f"✓ Generated {len(csm_data['metrics'])} metrics, {len(csm_data['dimensions'])} dimensions, {len(csm_data['relationships'])} relationships")
    print(f"✓ Generated {len(bgo_data.get('intent_patterns', []))} intent patterns")
    print(f"✓ Written csm_enterprise.yaml and bgo.yaml")
    print("=" * 60)

if __name__ == "__main__":
    main()