import json
import decimal
from core.pipeline import analytics_pipeline
from core.intent_processor import validate_question, NotAQuestionError
from core.lineage_tracker import explain as explain_lineage

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal): return float(obj)
    raise TypeError

def ask_database(query: str):
    query = validate_question(query)
    output = analytics_pipeline.invoke(query)

    # Check for DB execution error
    if output.get("execution_error"):
        print(f"\n❌ Database Error: {output['execution_error']}\n")
        return output

    print(f"\nGenerated SQL:\n{output['sql']}\n")

    # Display results
    results = output.get("results", {})
    rows = results.get("rows", [])
    print(f"Data Output ({results.get('row_count', 0)} rows):",
          json.dumps(rows, indent=2, default=decimal_default))

    # Display insight
    insight = output.get("insight", "")
    if insight and insight != "INSUFFICIENT":
        print(f"\n💡 Insight: {insight}")

    # Display viz recommendation
    viz = output.get("viz_spec", {})
    if viz.get("chart_type"):
        print(f"\n📊 Suggested Chart: {viz['chart_type']} — {viz.get('title', '')}")

    # Display lineage
    lineage = output.get("lineage")
    if lineage:
        print(f"\n{explain_lineage(lineage)}")

    # Display audit ID
    if output.get("audit_id"):
        print(f"\n🔖 Audit ID: {output['audit_id']}")

    return output

if __name__ == "__main__":
    while True:
        try:
            question = input("Ask your question (or 'exit' to quit): ").strip()
            if question.lower() in ("exit", "quit", "q"): break
            ask_database(question)
        except NotAQuestionError as e:
            print(f"\n[input error] {e}\n")
        except KeyboardInterrupt:
            break