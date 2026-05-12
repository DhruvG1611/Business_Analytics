"""
result_normalizer.py
--------------------
Converts raw list[dict] query results into a standardized Dataset dict
that serves as the data contract for all downstream pipeline components.

No LLM call. No external dependencies beyond standard Python.
"""
import statistics
from decimal import Decimal
from datetime import datetime, date


def _detect_type(values: list) -> str:
    """Infer column type from actual non-null values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "string"

    # Check date first (datetime is subclass of date)
    if all(isinstance(v, (datetime, date)) for v in non_null):
        return "date"

    # Check int — must be int type and not bool (bool is subclass of int)
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "int"

    # Check float / Decimal
    if all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in non_null):
        return "float"

    return "string"


def _compute_stats(values: list) -> dict:
    """Compute min, max, mean, null_count for a numeric column."""
    non_null = [float(v) for v in values if v is not None]
    null_count = len(values) - len(non_null)

    if not non_null:
        return {"min": None, "max": None, "mean": None, "null_count": null_count}

    return {
        "min": min(non_null),
        "max": max(non_null),
        "mean": round(statistics.mean(non_null), 4),
        "null_count": null_count,
    }


def normalize(raw_rows: list[dict]) -> dict:
    """
    Normalize raw query results into a Dataset dict.

    Returns:
        {
            "columns":      [{"name": str, "detected_type": str}],
            "rows":         list[dict],
            "stats":        {col_name: {min, max, mean, null_count}},
            "row_count":    int,
            "display_hint": "single_value" | "tabular" | "empty"
        }
    """
    if not raw_rows:
        return {
            "columns": [],
            "rows": [],
            "stats": {},
            "row_count": 0,
            "display_hint": "empty",
        }

    # Detect column types from actual values
    col_names = list(raw_rows[0].keys())
    columns = []
    col_types = {}

    for col in col_names:
        values = [row.get(col) for row in raw_rows]
        detected = _detect_type(values)
        columns.append({"name": col, "detected_type": detected})
        col_types[col] = detected

    # Compute stats for numeric columns
    stats = {}
    for col in col_names:
        if col_types[col] in ("int", "float"):
            values = [row.get(col) for row in raw_rows]
            stats[col] = _compute_stats(values)

    # Determine display hint
    row_count = len(raw_rows)
    numeric_cols = [c for c in col_names if col_types[c] in ("int", "float")]

    if row_count == 0:
        display_hint = "empty"
    elif row_count == 1 and len(numeric_cols) == 1:
        display_hint = "single_value"
    else:
        display_hint = "tabular"

    return {
        "columns": columns,
        "rows": raw_rows,
        "stats": stats,
        "row_count": row_count,
        "display_hint": display_hint,
    }
