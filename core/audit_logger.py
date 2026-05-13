"""
audit_logger.py
---------------
Append-only JSON Lines audit log for every pipeline run.
Masks PII columns in SQL before writing.

File: audit_log.jsonl (project root, append-only — never overwrite).
"""
import json
import re
import uuid
from datetime import datetime, timezone

import config

_AUDIT_FILE = "audit_log.jsonl"


def _mask_pii(sql: str) -> str:
    """Replace PII column references in SQL with [REDACTED]."""
    masked = sql
    for col_ref in config.PII_COLUMNS:
        # Case-insensitive replacement of the column reference
        pattern = re.compile(re.escape(col_ref), re.IGNORECASE)
        masked = pattern.sub("[REDACTED]", masked)
    return masked


def log_run(state: dict) -> str:
    """
    Write one audit log entry and return the run_id.

    Reads from state: question, intent, sql, results, insight, viz_spec, execution_error.
    """
    run_id = uuid.uuid4().hex
    intent = state.get("intent", {})
    results = state.get("results", {})
    insight = state.get("insight", "")
    viz_spec = state.get("viz_spec", {})
    error = state.get("execution_error", None)

    # Determine if insight was successfully generated
    insight_generated = bool(
        insight
        and insight != "INSUFFICIENT"
        and insight != "Insight generation failed."
    )

    cost = state.get("_cost_estimate")
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": state.get("question", ""),
        "metric_key": intent.get("metric", ""),
        "final_sql": _mask_pii(state.get("sql", "")),
        "row_count": results.get("row_count", 0),
        "insight_generated": insight_generated,
        "error": error,
        "viz_chart_type": viz_spec.get("chart_type", ""),
        "cost": {
            "total_usd":          cost.total_cost_usd   if cost else None,
            "total_tokens":       cost.total_tokens      if cost else None,
            "prompt_tokens":      cost.prompt_tokens     if cost else None,
            "completion_tokens":  cost.completion_tokens if cost else None,
            "model":              cost.model             if cost else None,
            "estimated":          cost.estimated         if cost else None,
        }
    }

    with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return run_id


def query_audit(run_id: str) -> dict:
    """
    Look up an audit log entry by run_id.

    Reads audit_log.jsonl line by line and returns the matching entry.
    Raises ValueError if not found.
    """
    try:
        with open(_AUDIT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("run_id") == run_id:
                    return entry
    except FileNotFoundError:
        pass

    raise ValueError(f"Audit entry not found for run_id: {run_id}")
