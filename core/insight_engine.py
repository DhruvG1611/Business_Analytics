"""
insight_engine.py
-----------------
Generates a 2–4 sentence natural language insight from the numeric
aggregate stats of a normalized Dataset. Never sees raw row data.

Uses config.llm directly (not through the decomposition chain).
"""
import config
from langchain_core.messages import SystemMessage, HumanMessage


_SYSTEM_PROMPT = (
    "You are a concise data analyst. "
    "Interpret only the numbers provided. "
    "Never invent context or reference data you weren't given. "
    "Respond in 2–4 sentences. "
    "If fewer than 2 numeric data points exist, respond with only the word INSUFFICIENT."
)


def generate_insight(dataset: dict, question: str) -> str:
    """
    Generate a short insight from the Dataset's aggregate stats.

    Args:
        dataset:  Normalized Dataset dict (from result_normalizer).
        question: The original user question for context.

    Returns:
        A plain string — the insight text, or "INSUFFICIENT".
    """
    stats = dataset.get("stats", {})
    row_count = dataset.get("row_count", 0)

    # Guard: skip LLM if fewer than 2 numeric data points
    if len(stats) < 2:
        # Also check: if there's only 1 stat column but it has enough rows,
        # we still consider it valid (2+ data points = 2+ rows with 1 numeric col)
        if len(stats) == 1 and row_count >= 2:
            pass  # proceed — there are enough data points
        else:
            return "INSUFFICIENT"

    # Build compact stats summary — never send raw rows
    summary_lines = [f"row_count: {row_count}"]
    for col_name, col_stats in stats.items():
        summary_lines.append(
            f'column "{col_name}": '
            f"min={col_stats['min']}, max={col_stats['max']}, mean={col_stats['mean']}"
        )
    stats_summary = "\n".join(summary_lines)

    user_message = (
        f'The user asked: "{question}"\n\n'
        f"Here are the aggregate statistics from the query results:\n{stats_summary}\n\n"
        f"Provide a brief analytical insight based on these numbers."
    )

    try:
        response = config.llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        return response.content.strip()
    except Exception as e:
        print(f"  [InsightEngine] LLM call failed: {e}")
        return "Insight generation failed."
