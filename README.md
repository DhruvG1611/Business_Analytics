# Acceleron — Natural Language to SQL Analytics Pipeline

Acceleron is a semantic NLP-to-SQL pipeline that converts natural language business questions into executable MySQL queries against the Sakila DVD rental database. It solves the problem of bridging the gap between how business users think about data ("What is the total revenue by film category?") and the multi-table SQL joins required to answer those questions. The system uses a YAML-defined semantic layer (CSM + BGO), FAISS-based retrieval-augmented generation for intent grounding, an LLM for question decomposition, a graph-based join resolver that computes shortest paths through the schema, and a self-healing metric fallback that auto-generates missing metric definitions. After SQL execution, the pipeline normalizes results, generates LLM-powered insights, recommends chart types, tracks full query lineage, and writes PII-masked audit logs.

---

## Architecture Overview

The pipeline transforms a user's natural language question into SQL and structured output through 11 sequential steps, each implemented as a LangChain `RunnableLambda` and chained via the `|` operator:

1. **RAG Context Retrieval** — The question is encoded into a vector using `sentence-transformers/all-MiniLM-L6-v2` and searched against two FAISS indexes: one for metrics/dimensions and one for intent patterns. The top-k results are scored. A pattern must clear a 0.82 cosine similarity threshold to be treated as a valid override. The retrieved context is serialized into a compact prompt string injected into the LLM's system context, scoped to only the relevant glossary entries rather than the full BGO dump.

2. **LLM Decomposition** — The user question, schema context (static JSON of all CSM metrics and dimensions built once at import time), and RAG context are sent to the Groq API (`llama-3.3-70b-versatile`, temperature=0) via a `ChatPromptTemplate`. The LLM returns a draft JSON intent with fields: `metric`, `dimensions`, `filters`, `sort`, `limit`, `mode`. A `JsonOutputParser` validates the output.

3. **Intent Parsing** — The `IntentParser` class runs a six-stage resolution pipeline on the draft intent: (a) query mode detection (simple/ranked/threshold/rank_per_group), (b) metric normalization through a 5-tier fuzzy lookup cascade (exact match → BGO alias → semantic synonym → `difflib.get_close_matches` → token overlap), (c) dimension inference from the question text when the LLM drops them, (d) safe RAG hint application (copies metric/dimensions/sort/limit from the best pattern, guarded by the 0.82 threshold and join-path protection), (e) dimension validation against the CSM dimension map, (f) irrelevant dimension filtering that prunes dimensions whose source table is not in the metric's join path, and (g) status filter injection for domain-specific conditions like `active = 1`.

4. **Metric Existence Check (Self-Healing)** — If the resolved metric key is not found in `config.csm["metrics"]`, the pipeline calls `auto_generate_metric()` which prompts the LLM to generate a CSM metric definition (compute expression, sources, join path). The generated definition is written to both `csm_enterprise.yaml` and `bgo.yaml` using `ruamel.yaml` to preserve formatting, then hot-reloaded into `config.csm`. The generated SQL is validated with a `LIMIT 0` test query. If validation fails, the user is prompted interactively for a manual definition. This step is a `RunnableLambda` between intent parsing and join resolution.

5. **Graph-Based Join Resolution** — `SchemaGraphResolver` builds a `networkx.DiGraph` from the CSM `relationships` section. Each FK relationship becomes a bidirectional weighted edge (self-joins weighted 10, others weighted 1). For metrics with an explicit `join_path`, the resolver walks the chain in order. For additional dimension tables not on the join path, it runs Dijkstra's shortest path algorithm to find the minimum-cost route from the base table, with tables outside the metric's join path treated as forbidden nodes. The output is a logical plan dict with `select`, `from`, `joins`, `group_by`, `filters`, `compute_where`, `sort`, `limit`, and `mode`.

6. **SQL Compilation** — The `sql_compiler` module converts the logical plan into executable MySQL. It supports four compilation modes: `simple` (SELECT/GROUP BY), `threshold` (adds HAVING), `ranked` (adds ORDER BY + LIMIT), and `rank_per_group` (wraps in a CTE with `RANK() OVER (PARTITION BY ...)`). Filter values are LOWER-compared for case insensitivity. Boolean columns emit numeric comparisons. The `compute_where` from the CSM metric is injected as a separate WHERE clause, never embedded inside the aggregate function.

7. **SQL Execution** — Runs `db.execute_query()` against MySQL via `mysql.connector`. On success, the raw `list[dict]` is added to pipeline state as `raw_rows`. On failure, the exception is caught and stored as `execution_error` in state — the pipeline continues to produce a complete audit log entry rather than crashing.

8. **Result Normalization** — The `result_normalizer` converts `raw_rows` into a typed `Dataset` dict: columns are type-detected (int/float/string/date) from actual values, numeric columns get min/max/mean/null_count stats, and a `display_hint` ("single_value"/"tabular"/"empty") is set based on result shape. This Dataset dict is the data contract for all downstream components.

9. **Insight Generation** — The `insight_engine` extracts only aggregate stats (never raw rows) and sends them to the LLM via a direct `SystemMessage`/`HumanMessage` invoke. The system prompt constrains the LLM to 2–4 sentences interpreting only the provided numbers. A guard skips the LLM call entirely if fewer than 2 numeric data points exist, returning `"INSUFFICIENT"`.

10. **Visualization Recommendation** — The `viz_recommender` applies rule-based heuristics with no LLM call. Rules are evaluated in priority order: single-value display, bar chart (1 string + 1 numeric), line chart (1 date + 1 numeric), pie chart (1 string with ≤6 distinct values + 1 numeric), or fallback table.

11. **Audit Logging** — The `audit_logger` writes one append-only JSON line to `audit_log.jsonl`. Before writing, it scans `config.PII_COLUMNS` (a frozenset built at startup by cross-referencing bgo.yaml `pii: true` flags with CSM `source`/`column` fields) and does case-insensitive `re.escape`-based replacement of PII column references with `[REDACTED]` in the SQL string. A `query_audit(run_id)` function allows retrieving any historical entry.

---

## File Map

| File | Module | Responsibility |
|------|--------|----------------|
| `main.py` | Entry point | CLI loop, calls `analytics_pipeline.invoke()`, displays SQL, results, insight, viz, lineage, audit ID |
| `config.py` | Configuration | Loads CSM + BGO YAML, initializes Groq LLM, builds decomposition chain, computes `PII_COLUMNS` frozenset |
| `db.py` | Database | `mysql.connector` wrapper: `execute_query()`, `execute_query_scalar()`, schema introspection helpers |
| `.env` | Environment | `GROQ_API_KEY`, `HF_HUB_DISABLE_PROGRESS_BARS` |
| `decomposition_prompt.txt` | LLM prompt | Template with `{metrics_list}`, `{dimensions_list}`, `{schema_context}`, `{bgo_context}` slots |
| `csm_enterprise.yaml` | Semantic layer | Canonical Semantic Model: metrics (compute, sources, join_path), dimensions (source, column, type), relationships |
| `bgo.yaml` | Business glossary | Business Glossary & Ontology: entity descriptions, synonyms, business rules, intent patterns, views, PII fields |
| `core/pipeline.py` | Pipeline | 11-step LangChain `RunnableLambda` chain, SQL cache, state threading |
| `core/rag_retriever.py` | RAG engine | FAISS index loading, `SentenceTransformer` encoding, `retrieve()` → `RetrievalResult` |
| `core/retriever_context.py` | RAG bridge | `retrieve_once()`, `build_rag_context()`, `extract_rag_hints()` with 0.82 threshold enforcement |
| `core/intent_parser.py` | Intent resolution | `IntentParser` class: mode detection, metric normalization, dimension inference, RAG hint application, dimension filtering |
| `core/intent_processor.py` | Input validation | `validate_question()`, `NotAQuestionError`, `normalize_intent()` |
| `core/metric_generator.py` | Self-healing | `auto_generate_metric()`: LLM-based metric generation, `write_metric_to_yaml()`: ruamel.yaml writer with hot-reload |
| `core/result_normalizer.py` | Normalization | `normalize()`: type detection, stats computation, display hint classification |
| `core/insight_engine.py` | Insight | `generate_insight()`: stats-only LLM invoke with INSUFFICIENT guard |
| `core/viz_recommender.py` | Visualization | `recommend_viz()`: rule-based chart spec (bar/line/pie/single_value/table) |
| `core/lineage_tracker.py` | Lineage | `build_lineage()`: provenance assembly, `explain()`: CLI renderer |
| `core/audit_logger.py` | Audit | `log_run()`: PII-masked JSONL append, `query_audit()`: lookup by run_id |
| `query_engine/graph_resolver.py` | Join resolution | `SchemaGraphResolver` (networkx DiGraph), Dijkstra shortest path, `rag_plus_plus_resolver()` → logical plan |
| `query_engine/sql_compiler.py` | SQL generation | `sql_compiler()`: logical plan → MySQL string (simple/threshold/ranked/rank_per_group modes) |
| `build_embeddings.py` | Offline | Builds FAISS indexes from CSM + BGO into `embeddings/` directory |
| `generate_csm_bgo.py` | Code generation | Introspects live DB schema, uses LLM to generate `csm_enterprise.yaml` and `bgo.yaml` from scratch |

---

## Key Concepts

### CSM (Canonical Semantic Model)
`csm_enterprise.yaml` is the single source of truth for what the system can measure and how. Each **metric** defines a `compute` expression (e.g. `SUM(payment.amount)`), the `sources` tables it reads from, an optional `join_path` (the ordered list of tables the graph resolver must traverse), and optional `compute_where` (a SQL predicate injected as a WHERE clause, separate from the aggregate). Each **dimension** maps a business concept to a physical `source` table and `column`, with a `type` classification (id/string/number/boolean/time). Each **relationship** defines a foreign key join condition between two tables. The CSM is loaded once at startup into `config.csm` and is the contract between the intent parser, graph resolver, and SQL compiler.

### BGO (Business Glossary & Ontology)
`bgo.yaml` provides the natural language layer over the CSM. It contains entity descriptions and synonyms (so the LLM can understand that "movie" means `film`), business rules (so the LLM knows `active = 1` means a current customer), `intent_patterns` (pre-built question→metric→dimensions templates used by the FAISS pattern index), and PII field declarations. The BGO is consumed by the RAG retriever (via embeddings) and by the LLM decomposition prompt.

### RAG++ Layer
The retrieval-augmented generation layer uses two FAISS indexes (built offline by `build_embeddings.py`): a **metric/dimension index** that finds the most likely metric for a question, and a **pattern index** that finds the closest pre-written intent template. The "plus plus" refers to two key design choices: (1) RAG results are used as *hints*, not gatekeepers — a low-scoring match is passed to the LLM as context but never forcefully overrides the LLM's output, and (2) high-confidence pattern matches (score ≥ 0.82) can directly populate the intent's dimensions, sort, limit, and mode fields, but only if the metric's join_path guard allows it.

### IntentParser
A six-stage deterministic pipeline that runs after the LLM's draft intent. It normalizes metric keys through a cascade (exact → alias → synonym → fuzzy → token overlap), infers missing dimensions from question keywords, applies RAG pattern hints with guards, filters out dimensions whose source table isn't reachable from the metric's join path, and injects domain-specific WHERE conditions. This is where the system corrects LLM hallucinations — if the LLM says `film_revenue` but the CSM key is `revenue_by_film`, the fuzzy matcher resolves it.

### SchemaGraphResolver
Builds a `networkx.DiGraph` where each table is a node and each FK relationship is a bidirectional weighted edge. Given a metric's base table and the set of tables required by dimensions and filters, it walks the metric's explicit `join_path` first, then uses Dijkstra's algorithm for any remaining tables. Tables outside the metric's join path are marked as forbidden nodes to prevent the shortest path from routing through unrelated tables. The output is a logical plan that the SQL compiler can directly translate.

### Self-Healing Metric Fallback
When a user asks about a metric that doesn't exist in the CSM, the pipeline doesn't fail. Instead, Step 3.5 detects the missing key and calls `auto_generate_metric()`, which prompts the LLM to produce a valid CSM definition (compute expression, sources, join path). The definition is written to both YAML files using `ruamel.yaml` (preserving formatting and comments), hot-reloaded into `config.csm`, and validated by running the generated SQL with `LIMIT 0`. If the LLM-generated definition produces invalid SQL, the user is prompted interactively to provide a manual definition.

### Result Normalization Contract
The `Dataset` dict returned by `result_normalizer.normalize()` is the standard data contract for all post-execution components. It contains typed `columns` (with detected types), the original `rows`, aggregate `stats` per numeric column, `row_count`, and a `display_hint`. This contract means that the insight engine, viz recommender, lineage tracker, and audit logger never need to re-inspect raw database output — they all operate on the same normalized structure.

---

## Data Flow — Pipeline State After Each Step

```
Step 1  (RAG Retrieve):     + question, schema_context, bgo_context, _rag_result
Step 2  (LLM Decompose):    + intent (draft JSON)
Step 3  (Intent Parse):     intent (resolved), question
Step 3.5 (Metric Check):    (passthrough, may hot-reload config.csm)
Step 4  (Join Resolve):     + logical_plan
Step 5  (SQL Compile):      + sql
Step 6  (SQL Execute):      + raw_rows, (+ execution_error on failure)
Step 7  (Normalize):        + results {columns, rows, stats, row_count, display_hint}
Step 8  (Insight):          + insight (string)
Step 9  (Viz Recommend):    + viz_spec {chart_type, x_axis, y_axis, title, render_config}
Step 10 (Lineage):          + lineage {question, metric_key, compute_expr, join_path, filters_applied, final_sql, row_count}
Step 11 (Audit):            + audit_id (UUID4 hex string)
```

Final output dict keys: `sql`, `logical_plan`, `intent`, `question`, `raw_rows`, `results`, `insight`, `viz_spec`, `lineage`, `audit_id`.

---

## Design Decisions

### Dijkstra Shortest Path for Join Resolution
The schema is a graph where tables are nodes and FK relationships are edges. Dijkstra's algorithm finds the minimum-weight path between any two tables, which corresponds to the minimum number of JOINs needed. Self-joins are weighted 10× to discourage accidental traversal through self-referencing FKs. The alternative — hard-coding join paths for every possible dimension-metric combination — doesn't scale. The graph approach makes new metrics and dimensions work automatically as long as the CSM relationships are defined.

### RAG as Hint, Not Gatekeeper
The RAG layer provides the best matching metric and pattern to the LLM as context, but never forces a metric selection. A 0.72 similarity score might be the best match but still wrong — the LLM's broad language understanding is better at disambiguating edge cases. Only at 0.82+ does the pattern's dimensions get applied, and even then, metrics with explicit `join_path` entries are protected from dimension override.

### YAML as the Semantic Layer Instead of a Database
The CSM and BGO are YAML files, not database tables. This was chosen because: (1) the semantic model changes infrequently and version control (git diff) is more useful than migration scripts, (2) the self-healing metric generator needs to append entries programmatically while preserving human-written comments and formatting (`ruamel.yaml`), (3) YAML is human-readable and editable by non-engineers, and (4) there's no need for concurrent writes or transactional guarantees.

### ruamel.yaml for Writes
Standard `pyyaml` strips all comments and reorders keys on dump. `ruamel.yaml` preserves YAML comments, key ordering, and block scalar styles. This is critical for the self-healing layer — when `auto_generate_metric()` appends a new metric, the rest of the file (including hand-written comments like `# ════ REVENUE METRICS ════`) must remain intact.

### Groq Instead of Local Ollama
The project originally used `ChatOllama` with a local `llama3` instance. This was replaced with `ChatGroq` (`llama-3.3-70b-versatile`) for three reasons: (1) the 70B model produces significantly better metric resolution and JSON output than the 7B local model, (2) Groq's inference speed (~200 tokens/s) is faster than local Ollama on consumer hardware, (3) the API requires only a key swap in `config.py` with no changes to the LangChain chain structure.

### Append-Only Audit Log
`audit_log.jsonl` is append-only by design. Each pipeline run adds one JSON line. The file is never truncated or overwritten. This ensures that audit history is immutable — no pipeline error or crash can destroy prior entries. The PII masking happens before write: column references matching `config.PII_COLUMNS` (cross-referenced from bgo.yaml `pii: true` flags and CSM `source`/`column` pairs) are replaced with `[REDACTED]` in the SQL string only.

---

## Known Limitations and Edge Cases

1. **Dimension Leakage (Fixed)** — The LLM decomposition step sometimes emits film-level dimensions (`film.title`, `film_id`) for queries that should be category-level. This was fixed by adding `_filter_irrelevant_dimensions` in `IntentParser`, which prunes dimensions whose source table is not in the metric's join path. A secondary guard in `_apply_safe_rag_hints` strips film-table dimensions when the question mentions "category" but not "film".

2. **Pattern Threshold Not Enforced (Fixed)** — The RAG pattern threshold was originally set too low (0.72) and used `>` instead of `>=`, causing a pattern scoring 0.745 to inject dimensions from a "best earning film per category" template into an unrelated query. This was fixed by setting the threshold to 0.82 with `>=` comparison.

3. **PII Masking Produced `.` (Fixed)** — The `PII_COLUMNS` frozenset was originally built from `bgo.yaml` dimension entries, which have `pii: true` but lack `source`/`column` fields (those live in the CSM). This produced a single entry `"."` (from empty source + empty column concatenated with a dot), which replaced every dot-separated `table.column` reference in the SQL with `[REDACTED]`. Fixed by cross-referencing the bgo.yaml `pii` flag with the CSM's `source`/`column` fields and filtering out empty entries.

4. **Hot-Reload Stale Imports (Fixed)** — `graph_resolver.py` and `sql_compiler.py` originally used `from config import csm` at import time, holding a local copy. When `metric_generator.py` updated `config.csm`, the graph resolver still used the stale copy. Fixed by changing to `import config` and referencing `config.csm` everywhere.

5. **Interactive `input()` in Pipeline** — The self-healing metric fallback uses `input()` for CLI recovery. This works in the current CLI setup but will break in any HTTP/async deployment. Must be replaced with an exception-based or callback flow for production.

6. **CTE Metrics** — Metrics like `customer_churn_rate` require Common Table Expressions. The CSM supports `is_cte_metric` and `cte_definition` fields, but the SQL compiler does not currently compile these — they remain as manual SQL hints in `notes` fields.

7. **Single-Database Scope** — The system is hardcoded to the Sakila database. The CSM, BGO, embeddings, and decomposition prompt are all Sakila-specific.

---

## How to Run

### Prerequisites

- Python 3.11+
- MySQL 5.7+ with the [Sakila sample database](https://dev.mysql.com/doc/sakila/en/) loaded
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone <repo-url> && cd acceleronSoln
pip install langchain-core langchain-groq mysql-connector-python \
            ruamel.yaml python-dotenv faiss-cpu sentence-transformers \
            networkx numpy
```

### Configuration

Create `.env` in the project root:
```
GROQ_API_KEY=gsk_your_key_here
HF_HUB_DISABLE_PROGRESS_BARS=1
```

Verify MySQL credentials in `db.py` and `config.py` (default: `localhost:3306`, user `root`, no password, database `sakila`).

### Build Embeddings (One-Time)

```bash
python build_embeddings.py
```

This creates FAISS indexes in `embeddings/` from the current CSM and BGO.

### Run

```bash
python main.py
```

Type a question at the prompt. The output includes: the generated SQL, query results, an LLM-generated insight, a chart type recommendation, a full lineage trace, and an audit ID.

### Regenerate CSM + BGO from Schema (Optional)

```bash
python generate_csm_bgo.py
```

Back up the existing files, introspects the live database, and regenerates both YAML files using LLM enrichment. Cached in `llm_cache.json`.

---

## How to Extend

### Adding a New Metric Manually

1. Add the metric entry to `csm_enterprise.yaml` under `metrics:`:
   ```yaml
   my_new_metric:
     compute: SUM(payment.amount)
     sources: [payment, rental, customer]
     label: My New Metric
     join_path: [payment, rental, customer]
   ```
2. Add a corresponding entry to `bgo.yaml` under `metrics:` (label, synonyms, description, calculation).
3. Add 2–3 intent patterns to `bgo.yaml` under `intent_patterns:` for the questions this metric answers.
4. Re-run `python build_embeddings.py` to update the FAISS indexes.

### Letting the Self-Healing Layer Generate It

Ask a question that implies the metric. If the IntentParser resolves the metric key but it's not in the CSM, the pipeline will:
1. Prompt the LLM to generate a definition.
2. Write it to both YAML files via `ruamel.yaml`.
3. Validate with `LIMIT 0`.
4. Hot-reload `config.csm`.
5. If validation fails, prompt you interactively for a manual definition.

No embeddings rebuild is needed for the pipeline to use it immediately, but rebuild embeddings later for RAG to find it.

### Adding a New Database

1. Update `db.py` and `config.py` with new connection credentials.
2. Run `python generate_csm_bgo.py` to introspect the new schema and auto-generate both YAML files.
3. Review and refine the generated YAML — the LLM-generated descriptions and synonyms may need domain-specific tuning.
4. Update `decomposition_prompt.txt` if the domain vocabulary is significantly different.
5. Run `python build_embeddings.py` to create FAISS indexes for the new schema.
6. Run `python main.py` and test with domain-appropriate questions.