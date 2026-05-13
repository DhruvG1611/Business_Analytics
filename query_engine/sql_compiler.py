"""
sql_compiler.py
---------------
Converts a logical plan dict into executable MySQL/MariaDB SQL.

Returns (sql_string, params_tuple) — the caller must pass both to db.execute_query().
Filter values use %s placeholders to prevent SQL injection.

Supported modes:
  simple        — SELECT ... FROM ... WHERE ... GROUP BY ...
  threshold     — adds HAVING clause
  ranked        — adds ORDER BY result + LIMIT
  rank_per_group — wraps in a CTE with RANK() OVER (PARTITION BY ...)
                   and filters to rank = 1 (or top N)
"""
import config


def _build_where_clauses(filters: list, compute_where: str | None) -> tuple[list[str], list]:
    """
    Merges compute_where (raw SQL from CSM) with user dimension filters.
    Returns (clause_templates, params) — templates use %s for filter values.
    """
    clauses = []
    params  = []

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
            clauses.append(f"{col_ref} = %s")
            params.append(int(values[0]))
            continue

        # Parameterized filter — case-insensitive match via LOWER()
        if len(values) > 1:
            placeholders = ", ".join(["%s"] * len(values))
            clauses.append(f"LOWER({col_ref}) IN ({placeholders})")
            params.extend(str(v).lower() for v in values)
        else:
            clauses.append(f"LOWER({col_ref}) = %s")
            params.append(str(values[0]).lower())

    return clauses, params


def _build_base_query(plan: dict) -> tuple[str, list]:
    """Builds the inner SELECT…GROUP BY block used by all modes.
    Returns (sql_string, params_list)."""
    select_items = plan.get("select", [])
    sql = f"SELECT {', '.join(select_items)}\nFROM {plan['from']}"

    if plan.get("joins"):
        sql += "\n" + "\n".join(plan["joins"])

    clauses, params = _build_where_clauses(
        plan.get("filters", []),
        plan.get("compute_where"),
    )
    if clauses:
        sql += f"\nWHERE {' AND '.join(clauses)}"

    if plan.get("group_by"):
        sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    return sql, params


def _compile_simple(plan: dict) -> tuple[str, list]:
    return _build_base_query(plan)


def _compile_threshold(plan: dict) -> tuple[str, list]:
    sql, params = _build_base_query(plan)
    threshold = plan["having_threshold"]
    op_map = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<", "eq": "="}
    sql_op = op_map.get(threshold["op"], ">=")
    sql += f"\nHAVING result {sql_op} {threshold['val']}"
    return sql, params


def _compile_ranked(plan: dict) -> tuple[str, list]:
    sql, params = _build_base_query(plan)
    if plan.get("sort") == "desc":
        sql += "\nORDER BY result DESC"
    elif plan.get("sort") == "asc":
        sql += "\nORDER BY result ASC"
    if plan.get("limit"):
        sql += f"\nLIMIT {plan['limit']}"
    return sql, params


def _compile_rank_per_group(plan: dict) -> tuple[str, list]:
    """
    Emits a CTE using RANK() OVER (PARTITION BY <group_col> ORDER BY result DESC/ASC)
    then selects rows where rnk <= limit.
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

    clauses, params = _build_where_clauses(
        plan.get("filters", []),
        plan.get("compute_where"),
    )
    if clauses:
        inner_sql += f"\nWHERE {' AND '.join(clauses)}"

    if plan.get("group_by"):
        inner_sql += f"\nGROUP BY {', '.join(plan['group_by'])}"

    sql = f"WITH ranked AS (\n{inner_sql}\n)\nSELECT * FROM ranked WHERE rnk <= {limit_n}"
    return sql, params


def _compile_set_difference(plan: dict) -> tuple[str, list]:
    """
    Emits a query with a WHERE ... NOT IN (SELECT ...) subquery.
    """
    base_sql, base_params = _build_base_query(plan)
    
    exclude_sq = plan.get("exclude_subquery")
    if not exclude_sq:
        return base_sql, base_params
        
    sub_clauses, sub_params = _build_where_clauses(
        exclude_sq.get("filters", []),
        None
    )
    
    sub_sql = f"SELECT {', '.join(exclude_sq['select'])}\nFROM {exclude_sq['from']}"
    if exclude_sq.get("joins"):
        sub_sql += "\n" + "\n".join(exclude_sq["joins"])
    if sub_clauses:
        sub_sql += f"\nWHERE {' AND '.join(sub_clauses)}"
        
    # Append NOT IN clause
    exclude_col = plan.get("exclude_col", "film_id")
    
    # Check if WHERE already exists in base_sql
    if "\nWHERE" in base_sql:
        sql = base_sql + f"\nAND {exclude_col} NOT IN (\n{sub_sql}\n)"
    else:
        sql = base_sql + f"\nWHERE {exclude_col} NOT IN (\n{sub_sql}\n)"
        
    return sql, base_params + sub_params


def sql_compiler(plan: dict) -> tuple[str, list]:
    """Compile a logical plan into (sql_string, params_list)."""
    print(f"  [DEBUG sql_compiler] mode={plan.get('mode')!r}")
    mode = plan.get("mode", "simple")

    if mode == "rank_per_group":
        return _compile_rank_per_group(plan)
    elif mode == "set_difference":
        return _compile_set_difference(plan)
    elif plan.get("having_threshold"):
        return _compile_threshold(plan)
    elif plan.get("sort") and plan.get("limit"):
        return _compile_ranked(plan)
    elif plan.get("sort"):
        # Sort without limit (e.g. "sort by revenue")
        sql, params = _build_base_query(plan)
        sql += f"\nORDER BY result {'DESC' if plan['sort'] == 'desc' else 'ASC'}"
        return sql, params
    else:
        return _compile_simple(plan)