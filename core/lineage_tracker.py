"""
lineage_tracker.py
------------------
Assembles a provenance record from pipeline state.
No LLM call, no DB call — reads only from existing state and config.csm.
"""
import config


def build_lineage(state: dict) -> dict:
    """
    Build a lineage record from the current pipeline state.

    Reads: question, intent, logical_plan, sql, results
    """
    intent = state.get("intent", {})
    metric_key = intent.get("metric", "")

    # Look up the compute expression from the CSM
    metric_node = config.csm.get("metrics", {}).get(metric_key, {})
    compute_expr = metric_node.get("compute", "")

    # Extract join path from logical plan
    logical_plan = state.get("logical_plan", {})
    join_path = logical_plan.get("joins", [])

    # Extract results info
    results = state.get("results", {})
    row_count = results.get("row_count", 0)

    return {
        "question": state.get("question", ""),
        "metric_key": metric_key,
        "compute_expr": compute_expr,
        "join_path": join_path,
        "filters_applied": intent.get("filters", []),
        "final_sql": state.get("sql", ""),
        "row_count": row_count,
    }


def explain(lineage: dict) -> str:
    """
    Render a lineage record as a human-readable multi-line string for CLI display.
    """
    joins = lineage.get("join_path", [])
    if joins:
        # Extract table names from join clauses like "LEFT JOIN rental ON ..."
        join_display = " → ".join(
            j.replace("LEFT JOIN ", "").split(" ON ")[0].strip()
            for j in joins
            if isinstance(j, str)
        )
    else:
        join_display = "(none)"

    filters = lineage.get("filters_applied", [])
    if filters:
        filter_display = ", ".join(
            f"{f.get('col_key', '?')} {f.get('op', '=')} {f.get('val', f.get('vals', '?'))}"
            for f in filters
        )
    else:
        filter_display = "(none)"

    sql_display = lineage.get("final_sql", "")
    # Truncate long SQL for display
    if len(sql_display) > 120:
        sql_display = sql_display[:117] + "..."

    lines = [
        "──── Query Lineage ────",
        f"Question:  {lineage.get('question', '')}",
        f"Metric:    {lineage.get('metric_key', '')}",
        f"Compute:   {lineage.get('compute_expr', '')}",
        f"Joins:     {join_display}",
        f"Filters:   {filter_display}",
        f"SQL:       {sql_display}",
        f"Rows:      {lineage.get('row_count', 0)}",
        "────────────────────────",
    ]
    return "\n".join(lines)
