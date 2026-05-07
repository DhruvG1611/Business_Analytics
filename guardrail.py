"""
sql_guardrail.py
----------------
Validates generated SQL against the CSM schema BEFORE execution.
Catches bad syntax, hallucinated tables/columns, injection attempts,
and dangerous mutation statements.

Usage — drop into your connector and call inside ask_database():

    errors = validate_sql(final_sql, plan, csm)
    if errors:
        print("[GUARDRAIL] SQL rejected:")
        for e in errors:
            print(f"  - {e}")
        return []
"""

import re


# ---------------------------------------------------------------------------
# Allowed tables and columns derived directly from CSM at runtime
# ---------------------------------------------------------------------------

def _build_allowlist(csm: dict) -> dict:
    """
    Build a whitelist of allowed tables and their columns from the CSM.

    Returns:
        {
            "tables":  {"employees", "departments", "tasks", "projects"},
            "columns": {"employees.emp_name", "tasks.task_status", ...}
        }
    """
    tables  = set()
    columns = set()

    for dim in csm.get("dimensions", {}).values():
        source = dim.get("source", "")
        col    = dim.get("column", "")
        if source:
            tables.add(source.lower())
        if source and col:
            # strip any SQL function wrappers e.g. YEAR(tasks.created_at)
            bare = re.sub(r"[A-Z_]+\(([^)]+)\)", r"\1", col)
            columns.add(f"{source.lower()}.{bare.lower()}")

    for metric in csm.get("metrics", {}).values():
        for src in metric.get("sources", []):
            tables.add(src.lower())

    return {"tables": tables, "columns": columns}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_mutation(sql: str) -> list[str]:
    """Reject any non-SELECT statement."""
    stripped = sql.strip().lstrip("(").upper()
    mutation_keywords = ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
                         "ALTER", "CREATE", "REPLACE", "GRANT", "REVOKE")
    for kw in mutation_keywords:
        if stripped.startswith(kw) or re.search(rf"\b{kw}\b", stripped):
            return [f"Mutation keyword '{kw}' is not allowed."]
    return []


def _check_starts_with_select(sql: str) -> list[str]:
    """SQL must begin with SELECT (after optional whitespace/comments)."""
    clean = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    clean = re.sub(r"--[^\n]*", "", clean).strip()
    if not clean.upper().startswith("SELECT"):
        return ["SQL does not begin with SELECT."]
    return []


def _check_injection_patterns(sql: str) -> list[str]:
    """Catch classic injection patterns."""
    errors = []
    patterns = [
        (r";\s*\w",           "Multiple statements (semicolon followed by keyword) detected."),
        (r"--",               "SQL comment '--' detected — possible injection."),
        (r"/\*.*?\*/",        "Block comment detected — possible injection."),
        (r"\bOR\b\s+1\s*=\s*1",  "OR 1=1 tautology detected."),
        (r"\bAND\b\s+1\s*=\s*1", "AND 1=1 tautology detected."),
        (r"SLEEP\s*\(",       "SLEEP() call detected — possible blind injection."),
        (r"BENCHMARK\s*\(",   "BENCHMARK() call detected — possible blind injection."),
        (r"UNION\s+SELECT",   "UNION SELECT detected — possible injection."),
        (r"LOAD_FILE\s*\(",   "LOAD_FILE() call detected."),
        (r"INTO\s+OUTFILE",   "INTO OUTFILE detected."),
        (r"xp_cmdshell",      "xp_cmdshell detected."),
    ]
    for pattern, message in patterns:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            errors.append(message)
    return errors


def _check_tables(sql: str, allowlist: dict) -> list[str]:
    """
    Verify every table referenced in FROM / JOIN clauses is in the CSM.
    Handles aliases: FROM employees e  or  JOIN tasks t ON ...
    """
    errors  = []
    allowed = allowlist["tables"]

    # Extract table names from FROM and JOIN clauses (ignore aliases)
    table_pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )
    for match in table_pattern.finditer(sql):
        table = match.group(1).lower()
        if table not in allowed:
            errors.append(f"Unknown table '{table}' — not in CSM schema.")

    return errors


def _check_columns(sql: str, allowlist: dict, plan: dict) -> list[str]:
    """
    Verify every table.column reference in the SQL exists in the CSM.
    Only checks dotted references (table.column) to avoid false positives
    on aliases and computed expressions.
    """
    errors  = []
    allowed = allowlist["columns"]

    dot_ref_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b")

    for match in dot_ref_pattern.finditer(sql):
        table  = match.group(1).lower()
        column = match.group(2).lower()
        ref    = f"{table}.{column}"

        # Skip ON clause join keys (they may use FK columns not in dimensions)
        # Skip known aggregate functions and aliases
        if column in ("count", "sum", "avg", "min", "max", "result"):
            continue

        if ref not in allowed:
            errors.append(f"Unknown column reference '{table}.{column}' — not in CSM schema.")

    return errors


def _check_select_has_result(sql: str) -> list[str]:
    """Ensure the SELECT includes a metric alias 'result'."""
    # Find everything between SELECT and FROM
    match = re.search(r"SELECT\s+(.*?)\s+FROM\b", sql, re.IGNORECASE | re.DOTALL)
    if match:
        select_clause = match.group(1)
        if "result" not in select_clause.lower():
            return ["SELECT clause is missing the metric alias 'result'."]
    return []


def _check_limit_is_safe(sql: str, max_rows: int = 10_000) -> list[str]:
    """Warn if LIMIT exceeds a safe threshold (protects against runaway queries)."""
    match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    if match:
        limit = int(match.group(1))
        if limit > max_rows:
            return [f"LIMIT {limit} exceeds the maximum allowed rows ({max_rows})."]
    return []


def _check_group_by_matches_select(sql: str) -> list[str]:
    """
    If GROUP BY is present, every non-aggregate SELECT column should appear
    in the GROUP BY clause (basic ONLY_FULL_GROUP_BY guard).
    """
    errors = []

    select_match   = re.search(r"SELECT\s+(.*?)\s+FROM\b",    sql, re.IGNORECASE | re.DOTALL)
    group_by_match = re.search(r"GROUP\s+BY\s+(.*?)(?:HAVING|ORDER|LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)

    if not select_match or not group_by_match:
        return errors

    select_cols  = [c.strip().lower() for c in select_match.group(1).split(",")]
    group_by_cols = [c.strip().lower() for c in group_by_match.group(1).split(",")]

    aggregate_pattern = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)

    for col in select_cols:
        if "as result" in col:
            continue                              # skip the metric alias
        if aggregate_pattern.search(col):
            continue                              # skip aggregate expressions
        if col not in group_by_cols:
            errors.append(
                f"SELECT column '{col}' is not in GROUP BY — may cause ONLY_FULL_GROUP_BY error."
            )

    return errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_sql(sql: str, plan: dict, csm: dict, max_rows: int = 10_000) -> list[str]:
    """
    Run all guardrail checks on a generated SQL string.

    Args:
        sql:      The SQL string produced by sql_compiler().
        plan:     The logical plan dict from rag_plus_plus_resolver().
        csm:      The loaded CSM dict (metrics + dimensions).
        max_rows: Maximum allowed LIMIT value (default 10,000).

    Returns:
        A list of error strings. Empty list means the SQL passed all checks.

    Example:
        errors = validate_sql(final_sql, logical_plan, csm)
        if errors:
            for e in errors:
                print(f"  [GUARDRAIL] {e}")
            return []   # abort execution
    """
    allowlist = _build_allowlist(csm)
    errors: list[str] = []

    errors += _check_mutation(sql)
    errors += _check_starts_with_select(sql)
    errors += _check_injection_patterns(sql)
    errors += _check_tables(sql, allowlist)
    errors += _check_columns(sql, allowlist, plan)
    errors += _check_select_has_result(sql)
    errors += _check_limit_is_safe(sql, max_rows)
    errors += _check_group_by_matches_select(sql)

    return errors