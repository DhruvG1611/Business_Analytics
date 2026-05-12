"""
sql_compiler.py
---------------
Converts a logical plan dict into executable MySQL/MariaDB SQL.

Supported modes:
  simple        — SELECT ... FROM ... WHERE ... GROUP BY ...
  threshold     — adds HAVING clause
  ranked        — adds ORDER BY result + LIMIT
  rank_per_group — wraps in a CTE with RANK() OVER (PARTITION BY ...)
                   and filters to rank = 1 (or top N)
"""
import config


def _build_where_clauses(filters: list, compute_where: str | None) -> list[str]:
    """
    Merges compute_where (raw SQL from CSM) with user dimension filters.
    compute_where goes first so it can't be accidentally AND-ed into wrong position.
    """
    clauses = []

    # Metric-level condition (e.g. return_date IS NULL for active rentals)
    if compute_where:
        clauses.append(compute_where)

    for f in filters:
        if f.get("is_aggregate") or f.get("is_conditional"):
            continue

        dim_node = config.csm["dimensions"].get(f.get("col_key", ""))
        if not dim_node:
            continue

        col_ref = f"{dim_node['source']}.{dim_node['column']}"
        values  = f.get("vals") or [f.get("val")]
        if not values or values == [None]:
            continue

        # Boolean columns (active=1/0) — emit as numeric, not string
        if dim_node.get("type") == "boolean":
            clauses.append(f"{col_ref} = {values[0]}")
            continue

        safe_vals = [str(v).replace("'", "''") for v in values]
        if len(safe_vals) > 1:
            in_parts = ", ".join(f"LOWER('{v}')" for v in safe_vals)
            clauses.append(f"LOWER({col_ref}) IN ({in_parts})")
        else:
            clauses.append(f"LOWER({col_ref}) = LOWER('{safe_vals[0]}')")

    return clauses


def _build_base_query(plan: dict) -> str:
    """Builds the inner SELECT…GROUP BY block used by all modes."""
    select_items = plan.get("select", [])
    sql = f"SELECT {', '.join(select_items)}\nFROM {plan['from']}"

    if plan.get("joins"):
        sql += "\n" + "\n".join(plan["joins"])

    where_clauses = _build_where_clauses(
        plan.get("filters", []),
        plan.get("compute_where"),
    )
    if where_clauses:
        sql += f"\nWHERE {' AND '.join(where_clauses)}"

    if plan.get("group_by"):
        sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    return sql


def _compile_simple(plan: dict) -> str:
    return _build_base_query(plan)


def _compile_threshold(plan: dict) -> str:
    sql = _build_base_query(plan)
    threshold = plan["having_threshold"]
    op_map = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<", "eq": "="}
    sql_op = op_map.get(threshold["op"], ">=")
    sql += f"\nHAVING result {sql_op} {threshold['val']}"
    return sql


def _compile_ranked(plan: dict) -> str:
    sql = _build_base_query(plan)
    if plan.get("sort") == "desc":
        sql += "\nORDER BY result DESC"
    elif plan.get("sort") == "asc":
        sql += "\nORDER BY result ASC"
    if plan.get("limit"):
        sql += f"\nLIMIT {plan['limit']}"
    return sql


def _compile_rank_per_group(plan: dict) -> str:
    """
    Emits a CTE using RANK() OVER (PARTITION BY <group_col> ORDER BY result DESC/ASC)
    then selects rows where rnk <= limit.

    Example output:
        WITH ranked AS (
          SELECT country.country, customer.first_name, SUM(payment.amount) AS result,
                 RANK() OVER (PARTITION BY country.country ORDER BY SUM(payment.amount) DESC) AS rnk
          FROM payment
          LEFT JOIN customer ON ...
          LEFT JOIN country ON ...
          GROUP BY country.country, customer.customer_id, customer.first_name
        )
        SELECT * FROM ranked WHERE rnk <= 1
    """
    partition_by  = plan.get("partition_by", [])
    order_by_expr = plan.get("order_by_expr", "result")
    sort_dir      = "DESC" if plan.get("sort") == "desc" else "ASC"
    limit_n       = plan.get("limit", 1)

    if not partition_by:
        # Degrade gracefully to ranked if no partition key was detected
        print("  [SQLCompiler] rank_per_group has no partition_by — degrading to ranked")
        return _compile_ranked(plan)

    partition_str = ", ".join(partition_by)
    window_fn     = f"RANK() OVER (PARTITION BY {partition_str} ORDER BY {order_by_expr} {sort_dir}) AS rnk"

    # Build inner SELECT with window function appended
    inner_select  = plan.get("select", []) + [window_fn]
    inner_sql     = f"SELECT {', '.join(inner_select)}\nFROM {plan['from']}"

    if plan.get("joins"):
        inner_sql += "\n" + "\n".join(plan["joins"])

    where_clauses = _build_where_clauses(
        plan.get("filters", []),
        plan.get("compute_where"),
    )
    if where_clauses:
        inner_sql += f"\nWHERE {' AND '.join(where_clauses)}"

    if plan.get("group_by"):
        inner_sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    return f"WITH ranked AS (\n{inner_sql}\n)\nSELECT * FROM ranked WHERE rnk <= {limit_n}"


def sql_compiler(plan: dict) -> str:
    mode = plan.get("mode", "simple")

    if mode == "rank_per_group":
        return _compile_rank_per_group(plan)
    elif plan.get("having_threshold"):
        return _compile_threshold(plan)
    elif plan.get("sort") and plan.get("limit"):
        return _compile_ranked(plan)
    elif plan.get("sort"):
        # Sort without limit (e.g. "sort by revenue")
        sql = _build_base_query(plan)
        sql += f"\nORDER BY result {'DESC' if plan['sort'] == 'desc' else 'ASC'}"
        return sql
    else:
        return _compile_simple(plan)