"""
viz_recommender.py
------------------
Rule-based chart type recommendation from a normalized Dataset.
No LLM. No external dependencies.

Decision rules (first match wins):
  1. row_count == 1 and 1 numeric col      → single_value
  2. 1 string col + 1 numeric col          → bar
  3. 1 date col + 1 numeric col            → line
  4. 1 string col (≤6 distinct) + 1 numeric → pie
  5. Everything else                        → table
"""


def recommend_viz(dataset: dict) -> dict:
    """
    Recommend a visualization type based on Dataset shape.

    Returns:
        {
            "chart_type":    str,
            "x_axis":        str | None,
            "y_axis":        str | None,
            "title":         str,
            "render_config": dict
        }
    """
    columns = dataset.get("columns", [])
    rows = dataset.get("rows", [])
    row_count = dataset.get("row_count", 0)

    # Classify columns by type
    string_cols = [c["name"] for c in columns if c["detected_type"] == "string"]
    numeric_cols = [c["name"] for c in columns if c["detected_type"] in ("int", "float")]
    date_cols = [c["name"] for c in columns if c["detected_type"] == "date"]

    # Rule 1: single value
    if row_count == 1 and len(numeric_cols) == 1:
        y = numeric_cols[0]
        return {
            "chart_type": "single_value",
            "x_axis": None,
            "y_axis": y,
            "title": y,
            "render_config": {},
        }

    # Rule 2: bar chart — exactly 1 string + 1 numeric
    if len(string_cols) == 1 and len(numeric_cols) == 1:
        x, y = string_cols[0], numeric_cols[0]
        return {
            "chart_type": "bar",
            "x_axis": x,
            "y_axis": y,
            "title": f"{y} by {x}",
            "render_config": {"orientation": "vertical"},
        }

    # Rule 3: line chart — exactly 1 date + 1 numeric
    if len(date_cols) == 1 and len(numeric_cols) == 1:
        x, y = date_cols[0], numeric_cols[0]
        return {
            "chart_type": "line",
            "x_axis": x,
            "y_axis": y,
            "title": f"{y} by {x}",
            "render_config": {},
        }

    # Rule 4: pie chart — 1 string col with ≤6 distinct values + 1 numeric
    if len(string_cols) >= 1 and len(numeric_cols) >= 1:
        # Find a string column with ≤6 distinct values
        for s_col in string_cols:
            distinct = set(row.get(s_col) for row in rows if row.get(s_col) is not None)
            if len(distinct) <= 6:
                y = numeric_cols[0]
                return {
                    "chart_type": "pie",
                    "x_axis": s_col,
                    "y_axis": y,
                    "title": f"{y} by {s_col}",
                    "render_config": {"max_slices": 6},
                }

    # Rule 5: fallback table
    return {
        "chart_type": "table",
        "x_axis": None,
        "y_axis": None,
        "title": "Query Results",
        "render_config": {},
    }
