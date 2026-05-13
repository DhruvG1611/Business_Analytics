# 1. What is the LLM model name used in decomposition_chain?
#    The model name used is "llama-3.3-70b-versatile" (configured in config.py via ChatGroq).

# 2. What fields does the pipeline state dict carry between steps?
#    The state dict accumulates and carries the following fields across the pipeline steps:
#    - question (initial input)
#    - schema_context, bgo_context, metrics_list, dimensions_list, _rag_result (added by _prepare_context)
#    - intent (added by decomposition_chain and refined by _parse_intent)
#    - logical_plan (added by _resolve_joins)
#    - sql, sql_params (added by _compile_sql)
#    - raw_rows, execution_error (added by _execute_sql)
#    - results (added by _normalize_results)
#    - insight (added by _generate_insight)
#    - viz_spec (added by _recommend_viz)
#    - lineage (added by _build_lineage_step)
#    - audit_id (added by _log_audit)

# 3. Where is the LLM first called (which step, which function)?
#    The LLM is first called in Step 2 of the pipeline via the `decomposition_chain`.
#    In pipeline.py, this occurs in the `analytics_pipeline` definition within the RunnableParallel block:
#    | RunnableParallel({"intent": decomposition_chain, ...})

# 4. Is there any existing token counting or cost tracking anywhere?
#    No, there is no existing token counting or cost tracking logic in pipeline.py, config.py, or rag_retriever.py.

from dataclasses import dataclass
import logging

@dataclass
class TokenEstimate:
    prompt_tokens:     int
    completion_tokens: int   # estimated, not actual
    total_tokens:      int
    model:             str
    estimated:         bool  # True if tiktoken unavailable

@dataclass
class CostEstimate:
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    input_cost_usd:    float
    output_cost_usd:   float
    total_cost_usd:    float
    model:             str
    estimated:         bool

class CostModel:
    # Approximate pricing per 1M tokens (Groq/Llama-3 estimates)
    PRICING = {
        "llama-3.3-70b-versatile":  {"input": 0.59,  "output": 0.79},
        "llama-3.1-8b-instant":     {"input": 0.05,  "output": 0.08},
        "mixtral-8x7b-32768":       {"input": 0.24,  "output": 0.24},
        "gemma2-9b-it":             {"input": 0.20,  "output": 0.20},
        "default":                  {"input": 0.59,  "output": 0.79},
    }

    def __init__(self, model: str):
        self.model = model
        self.prices = self.PRICING.get(model, self.PRICING["default"])

    def calculate(self, estimate: TokenEstimate) -> CostEstimate:
        input_cost = (estimate.prompt_tokens / 1_000_000) * self.prices["input"]
        output_cost = (estimate.completion_tokens / 1_000_000) * self.prices["output"]
        return CostEstimate(
            prompt_tokens     = estimate.prompt_tokens,
            completion_tokens = estimate.completion_tokens,
            total_tokens      = estimate.total_tokens,
            input_cost_usd    = input_cost,
            output_cost_usd   = output_cost,
            total_cost_usd    = input_cost + output_cost,
            model             = self.model,
            estimated         = estimate.estimated
        )

def format_cost(estimate: CostEstimate) -> str:
    """Returns a formatted string for console logging."""
    est_label = " (ESTIMATED)" if estimate.estimated else ""
    return (
        f"{estimate.model}{est_label}: "
        f"{estimate.total_tokens} tokens (~${estimate.total_cost_usd:.4f})"
    )

class TokenCounter:
    def __init__(self, model: str):
        self.model = model
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._exact = True
        except ImportError:
            logging.warning("tiktoken is unavailable. Using word-count approximation for tokens.")
            self._enc   = None
            self._exact = False

    def count(self, text: str) -> int:
        if self._exact:
            return len(self._enc.encode(text))
        return int(len(text.split()) * 1.33)

    def estimate(
        self,
        prompt: str,
        expected_completion_tokens: int = 300,
    ) -> TokenEstimate:
        pt = self.count(prompt)
        return TokenEstimate(
            prompt_tokens     = pt,
            completion_tokens = expected_completion_tokens,
            total_tokens      = pt + expected_completion_tokens,
            model             = self.model,
            estimated         = not self._exact,
        )

def estimate_pipeline_cost(
    question:       str,
    schema_context: str,
    bgo_context:    str,
    model:          str,
    expected_completion_tokens: int = 300,
) -> CostEstimate:
    """
    Estimates the cost of one pipeline LLM call (Step 2 only —
    the decomposition chain). Does not account for insight 
    generation (Step 8) yet.
    """
    # Reconstruct the approximate prompt the LLM will receive.
    # We include the primary context blocks injected by pipeline.py Step 1.
    full_prompt = "\n\n".join([
        f"Question: {question}",
        f"Schema: {schema_context}",
        f"Glossary: {bgo_context}",
    ])

    counter  = TokenCounter(model)
    estimate = counter.estimate(full_prompt, expected_completion_tokens)
    cost     = CostModel(model).calculate(estimate)
    return cost

class SessionCostTracker:
    def __init__(self):
        self._total_usd    = 0.0
        self._total_tokens = 0
        self._query_count  = 0
        self._history: list[CostEstimate] = []

    def record(self, estimate: CostEstimate):
        if not estimate: return
        self._total_usd    += estimate.total_cost_usd
        self._total_tokens += estimate.total_tokens
        self._query_count  += 1
        self._history.append(estimate)

    def summary(self) -> str:
        if not self._query_count:
            return "No queries run this session."
        avg = self._total_usd / self._query_count
        return (
            f"\n━━━━ Session Cost Summary ━━━━\n"
            f"  Queries run   : {self._query_count}\n"
            f"  Total tokens  : {self._total_tokens:,}\n"
            f"  Total cost    : ${self._total_usd:.4f} USD\n"
            f"  Avg per query : ${avg:.4f} USD\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def reset(self):
        self.__init__()

if __name__ == "__main__":
    # Test with the configured model from config.py
    model = "llama-3.3-70b-versatile"
    tc = TokenCounter(model)
    
    print("--- Token Estimation Test ---")
    result = tc.estimate("List the top 10 films by revenue")
    print(result)
    
    print("\n--- Cost Estimation Test ---")
    cost = estimate_pipeline_cost(
        question="Which film categories generate the most total revenue?",
        schema_context="metrics: revenue_by_category, dimensions: category.name",
        bgo_context="RELEVANT METRICS: revenue_by_category",
        model=model
    )
    print(cost)
