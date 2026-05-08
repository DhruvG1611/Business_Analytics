"""
prompt_builder.py
-----------------
Generates the schema-specific sections of the decomposition prompt at runtime
from the CSM and BGO, so the prompt never needs to be rewritten for a new DB.

Drop this into your project and replace the hardcoded sections in connector.py
as shown at the bottom of this file.
"""

import yaml


def build_selection_guide(csm: dict, glossary: dict) -> str:
    """
    Auto-generates the SELECTION GUIDE block from the CSM dimensions.

    For every dimension that appears in a relationship (i.e. a grouping axis),
    emits a line like:
      "per / by department / division / unit"  ->  dimensions: ["departments_dept_name"]

    Synonym phrases come from the BGO dimension synonyms so the guide stays
    consistent with what the LLM has already been told.
    """
    lines = ["SELECTION GUIDE:"]

    bgo_dim_synonyms = glossary.get("dimensions", {})
    relationships    = csm.get("relationships", {})

    # Collect the 'to' (parent) tables from all relationships — these are the
    # natural grouping axes ("per department", "per project", etc.)
    parent_tables: dict[str, str] = {}   # table_name -> first matching dimension key
    for rel in relationships.values():
        to_table = rel.get("to", "")
        if not to_table:
            continue
        # Find a dimension whose source is this table (prefer name-looking cols)
        for dim_key, dim_val in csm.get("dimensions", {}).items():
            if dim_val.get("source") == to_table and dim_val.get("type") == "string":
                if to_table not in parent_tables:
                    parent_tables[to_table] = dim_key
                break

    # Emit one line per parent table
    for table, dim_key in parent_tables.items():
        # Pull synonyms from BGO, fall back to the raw table name
        synonyms = bgo_dim_synonyms.get(dim_key, [table.rstrip("s")])
        # Take the 3 most natural-sounding ones for the guide
        short_syns = synonyms[:3]
        phrase = " / ".join(short_syns)
        lines.append(f'- "per / by {phrase}"  ->  dimensions: ["{dim_key}"]')

    # Ranking lines are always the same shape, just use the CSM to confirm
    # sort/limit are supported (they always are, but this keeps it explicit)
    lines.append('- "most / top / highest / max"  ->  sort: "desc", limit: 1')
    lines.append('- "least / lowest / fewest / min"  ->  sort: "asc", limit: 1')
    lines.append('- No grouping requested (total count)  ->  dimensions: []')

    return "\n".join(lines)


def build_few_shot_examples(csm: dict, glossary: dict) -> str:
    """
    Auto-generates the EXAMPLES block entirely from:
      - CSM metrics (to pick measure keys)
      - CSM dimensions (to pick dimension keys)
      - BGO intent_patterns (to drive the Q/A pairs)
      - BGO metric/dimension synonyms (to write natural-sounding questions)

    Falls back to a minimal set of inferred examples if the BGO has no
    intent_patterns.
    """
    lines = ["EXAMPLES:"]

    metrics    = csm.get("metrics", {})
    dimensions = csm.get("dimensions", {})
    patterns   = glossary.get("intent_patterns", [])
    m_syns     = glossary.get("metrics", {})
    d_syns     = glossary.get("dimensions", {})

    def first_synonym(key: str, pool: dict, fallback: str) -> str:
        syns = pool.get(key, [])
        return syns[0] if syns else fallback

    if patterns:
        for p in patterns:
            metric_key = p.get("metric", "")
            dims       = p.get("dimensions", [])
            filters    = p.get("filters", [])
            sort       = p.get("sort")
            limit      = p.get("limit", 1)

            # Validate keys exist in CSM before emitting
            if metric_key not in metrics:
                continue
            valid_dims = [d for d in dims if d in dimensions]

            # Build a natural-language question from the pattern + synonyms
            pattern_str = p.get("pattern", "")
            # Replace {entity} style placeholders with real table names
            entity_guess = metric_key.split("_")[0]   # e.g. "employees"
            q = pattern_str.replace("{entity}", entity_guess)

            # Build the intent JSON
            intent: dict = {
                "metric":     metric_key,
                "dimensions": valid_dims,
                "filters":    filters,
                "limit":      limit if sort else None,
                "sort":       sort,
            }

            intent_str = _format_intent(intent)
            lines.append(f'\nQ: "{q}"')
            lines.append(f'A: {{"intent": {intent_str}}}')

    else:
        # No BGO patterns — infer a minimal example set from the CSM directly
        for metric_key, metric_val in list(metrics.items())[:6]:
            # List example: metric + first string dimension from its source table
            source_table = metric_val.get("sources", [""])[0]
            name_dim = next(
                (k for k, v in dimensions.items()
                 if v.get("source") == source_table and v.get("type") == "string"),
                None
            )

            natural_q = first_synonym(metric_key, m_syns, f"list all {source_table}")
            intent: dict = {
                "metric":     metric_key,
                "dimensions": [name_dim] if name_dim else [],
                "filters":    [],
                "limit":      None,
                "sort":       None,
            }
            lines.append(f'\nQ: "{natural_q}"')
            lines.append(f'A: {{"intent": {_format_intent(intent)}}}')

    return "\n".join(lines)


def _format_intent(intent: dict) -> str:
    """Render an intent dict as a compact JSON-like string for the prompt."""
    dims    = intent.get("dimensions", [])
    filters = intent.get("filters", [])
    limit   = intent.get("limit")
    sort    = intent.get("sort")

    dims_str    = str(dims).replace("'", '"')
    filters_str = _format_filters(filters)

    return (
        f'{{"metric": "{intent["metric"]}", '
        f'"dimensions": {dims_str}, '
        f'"filters": {filters_str}, '
        f'"limit": {"null" if limit is None else limit}, '
        f'"sort": {"null" if sort is None else chr(34)+sort+chr(34)}}}'
    )


def _format_filters(filters: list) -> str:
    if not filters:
        return "[]"
    parts = []
    for f in filters:
        field  = f.get("field", "")
        op     = f.get("operator", "equals")
        values = f.get("values", ["<value>"])
        val    = values[0] if values else "<value>"
        parts.append(
            f'{{"field": "{field}", "operator": "{op}", "values": ["{val}"]}}'
        )
    return "[" + ", ".join(parts) + "]"


def build_full_prompt_template(csm: dict, glossary: dict) -> str:
    """
    Assembles the complete decomposition prompt template.

    The returned string still contains {metrics_list}, {dimensions_list},
    {schema_context}, and {bgo_context} as LangChain template variables —
    those are injected at runtime by the pipeline RunnableParallel.

    The SELECTION GUIDE and EXAMPLES are rendered statically from the CSM+BGO
    so they always match the actual schema.

    Call this once at startup (after loading CSM and BGO) and pass the result
    to ChatPromptTemplate.from_template().
    """
    selection_guide = build_selection_guide(csm, glossary)
    examples        = build_few_shot_examples(csm, glossary)

    return f"""You are a precise semantic parser for a business analytics system.
Translate the user's question into a JSON intent object using ONLY the provided
metrics and dimensions. Resolve synonyms using the Business Glossary below.

Available metrics (use exactly as written):
{{metrics_list}}

Available dimensions (use exactly as written):
{{dimensions_list}}

DATA MODEL CONTEXT (from CSM YAML):
{{schema_context}}

BUSINESS GLOSSARY & ONTOLOGY:
{{bgo_context}}

RULES:
1. Return ONLY valid JSON -- no markdown, no explanation.
2. "metric"     : pick ONE key from the metrics list. Use the METRIC SYNONYMS
                  section in the glossary to resolve informal phrasing.
3. "dimensions" : zero or more keys from the dimensions list. Use the DIMENSION
                  SYNONYMS section to resolve informal phrasing. Use the
                  ENTITY RELATIONSHIPS section to decide which dimension to
                  group by.
4. "filters"    : ONLY add when a SPECIFIC value is explicitly mentioned.
   Filter format: {{"field": "<dimension_key>", "operator": "equals", "values": ["<value>"]}}
   Supported operators: equals, notEquals, contains, gt, gte, lt, lte
5. "limit"      : integer or null.
6. "sort"       : "asc" | "desc" | null.

{selection_guide}

{examples}

QUESTION: {{question}}
RESPONSE:"""


# =============================================================================
# HOW TO USE IN connector.py
# =============================================================================
# Replace this in connector.py:
#
#   decomposition_chain = ChatPromptTemplate.from_template("""...""") | llm | JsonOutputParser()
#
# With this:
#
#   from prompt_builder import build_full_prompt_template
#
#   _PROMPT_TEMPLATE = build_full_prompt_template(csm, glossary)   # called once at startup
#
#   decomposition_chain = (
#       ChatPromptTemplate.from_template(_PROMPT_TEMPLATE) | llm | JsonOutputParser()
#   )
#
# That's it. The prompt now rebuilds itself from the CSM+BGO every time you
# run generate_csm_bgo.py, with no manual edits needed.
# =============================================================================
