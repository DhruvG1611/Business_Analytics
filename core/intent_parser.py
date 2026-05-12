"""intent_parser.py

Centralized intent resolution. Handles in order:
  1. Query mode detection (simple / ranked / threshold / rank_per_group)
  2. Metric normalization with dynamic Enum + 5-tier fuzzy lookup
  3. Dimension inference when LLM drops them
  4. Safe RAG hint application (copies metric, dimensions, partition_by,
     mode, sort, limit from matched pattern)
  5. Sakila-domain status filter injection (word-boundary safe)
"""
import re
import difflib
from enum import Enum
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOISE_TOKENS = {"row", "count", "by", "per", "total", "avg", "average"}

_BGO_ALIAS_MAP = {
    "film_in_stock":              "copies_per_film",
    "film_not_in_stock":          "copies_per_film",
    "rewards_report":             "revenue_by_customer",
    "get_customer_balance":       "unpaid_rental_count",
    "inventory_in_stock":         "copies_per_film",
    "inventory_held_by_customer": "active_rental_count",
    "customer_balance":           "unpaid_rental_count",
    "outstanding_balance":        "unpaid_rental_count",
    "unpaid_balance":             "unpaid_rental_count",
    "balance":                    "unpaid_rental_count",
    "sales_by_film_category":     "revenue_by_category",
    "sales_by_store":             "revenue_by_store",
    "actor_info":                 "films_per_actor",
    "customer_list":              "customer_row_count",
    "film_list":                  "film_row_count",
    "nicer_but_slower_film_list": "film_row_count",
    "staff_list":                 "staff_row_count",
}

_SEMANTIC_SYNONYMS = {
    "customer_spend_per_country":    "revenue_by_customer",
    "customer_spending_per_country": "revenue_by_customer",
    "spend_per_country":             "revenue_by_customer",
    "spending_by_country":           "revenue_by_customer",
    "top_spender_by_country":        "revenue_by_customer",
    "highest_spending_customer":     "revenue_by_customer",
    "customer_spend_per_customer":   "revenue_by_customer",
    "customer_lifetime_value":       "revenue_by_customer",
    "clv":                           "revenue_by_customer",
    "ltv":                           "revenue_by_customer",
    "unpaid_rentals":                "unpaid_rental_count",
    "rentals_unpaid":                "unpaid_rental_count",
    "outstanding_rentals":           "unpaid_rental_count",
    "customers_with_balance":        "unpaid_rental_count",
    "customers_outstanding":         "unpaid_rental_count",
    "revenue":                       "total_revenue",
    "total_revenue_all":             "total_revenue",
    "rental_count":                  "rental_row_count",
    "open_rentals":                  "active_rental_count",
    "current_rentals":               "active_rental_count",
    "checked_out":                   "active_rental_count",
}

_STATUS_MAP = {
    r"\bactive\b":   ("customer_active", "1"),
    r"\binactive\b": ("customer_active", "0"),
}

_PER_GROUP_PATTERNS = [
    r"\beach\b",
    r"\bper\s+(country|city|store|category|staff|actor|film|customer|genre)\b",
    r"\bby\s+(country|city|store|category|staff|actor|film|customer|genre)\b",
    r"\bfor\s+each\b",
    r"\bin\s+each\b",
    r"\bper\s+group\b",
    r"\bbreakdown\b",
]

_SUBJECT_DIM_HINTS = {
    "country":  ["country_country"],
    "city":     ["city_city"],
    "store":    ["store_store_id"],
    "category": ["category_name"],
    "actor":    ["actor_first_name", "actor_last_name"],
    "film":     ["film_title"],
    "customer": ["customer_first_name", "customer_last_name"],
    "staff":    ["staff_first_name", "staff_last_name"],
    "language": ["language_name"],
    "rating":   ["film_rating"],
}

# Name-only columns — never valid as PARTITION BY keys
_NAME_DIMS = {
    "customer_first_name", "customer_last_name",
    "actor_first_name",    "actor_last_name",
    "staff_first_name",    "staff_last_name",
}

# ---------------------------------------------------------------------------
# IntentParser
# ---------------------------------------------------------------------------

class IntentParser:
    def __init__(self, csm: dict, glossary: dict):
        self.csm = csm
        self.glossary = glossary
        self.metric_keys = list(csm.get("metrics", {}).keys())
        self.metric_map = {k.lower(): k for k in self.metric_keys}
        self.dimension_map = {k.lower(): k for k in csm.get("dimensions", {}).keys()}
        self.MetricEnum = Enum("MetricEnum", {k.upper(): k for k in self.metric_keys})

    def process_intent(self, raw_intent: dict, question: str, rag_hints: dict) -> dict:
        intent = raw_intent.get("intent", raw_intent).copy()
        q_lower = question.lower()

        intent = self._detect_mode(intent, q_lower)
        intent = self._normalize_metric(intent)
        intent = self._infer_dimensions(intent, q_lower)
        intent = self._apply_safe_rag_hints(intent, q_lower, rag_hints)
        intent = self._validate_dimensions(intent)
        intent = self._filter_irrelevant_dimensions(intent)
        intent = self._inject_status_filter(intent, q_lower)

        return {"intent": intent}

    # -----------------------------------------------------------------------
    # Step 1 — Mode detection
    # -----------------------------------------------------------------------

    def _detect_mode(self, intent: dict, q_lower: str) -> dict:
        having_patterns = [
            (r"at least\s+(\d+)",   "gte"),
            (r"more than\s+(\d+)",  "gt"),
            (r"over\s+(\d+)",       "gt"),
            (r"at most\s+(\d+)",    "lte"),
            (r"fewer than\s+(\d+)", "lt"),
            (r"less than\s+(\d+)",  "lt"),
        ]
        for pattern, op in having_patterns:
            m = re.search(pattern, q_lower)
            if m:
                intent["having_threshold"] = {"op": op, "val": int(m.group(1))}
                intent["sort"], intent["limit"] = None, None
                intent["mode"] = "threshold"
                return intent

        is_per_group = any(re.search(p, q_lower) for p in _PER_GROUP_PATTERNS)
        is_ranking   = any(kw in q_lower for kw in
                           ["most", "top", "highest", "best", "least", "lowest", "bottom"])

        if is_ranking and is_per_group:
            intent["mode"]  = "rank_per_group"
            intent["sort"]  = "desc" if any(kw in q_lower for kw in
                                            ["most", "top", "highest", "best"]) else "asc"
            intent["limit"] = self._extract_explicit_n(q_lower) or 1
            return intent

        if is_ranking:
            n = self._extract_explicit_n(q_lower) or 1
            intent["sort"]  = "desc" if any(kw in q_lower for kw in
                                            ["most", "top", "highest", "best"]) else "asc"
            intent["limit"] = n
            intent["mode"]  = "ranked"
            return intent

        intent.setdefault("mode", "simple")
        return intent

    def _extract_explicit_n(self, q_lower: str) -> int | None:
        m = re.search(r"\b(?:top|bottom|first|last)\s+(\d+)\b", q_lower)
        if m: return int(m.group(1))
        m = re.search(r"\b(\d+)\s+(?:customers?|films?|stores?|staff|actors?|categories)\b", q_lower)
        return int(m.group(1)) if m else None

    # -----------------------------------------------------------------------
    # Step 2 — Metric normalization
    # -----------------------------------------------------------------------

    def _normalize_metric(self, intent: dict) -> dict:
        raw = intent.get("metric", "")
        if isinstance(raw, Enum):
            raw = raw.value
        intent["metric"] = self._fuzzy_metric_lookup(raw)
        return intent

    def _fuzzy_metric_lookup(self, raw_metric: str) -> str:
        if isinstance(raw_metric, dict):
            # If the LLM returned a dict, try to grab the string from common keys, or fallback to casting it
            raw_metric = raw_metric.get("name", raw_metric.get("value", str(raw_metric)))
        elif not isinstance(raw_metric, str):
            # Catch any other weird types (like lists or None)
            raw_metric = str(raw_metric) if raw_metric is not None else ""
        key = raw_metric.lower()

        if key in _BGO_ALIAS_MAP:      return _BGO_ALIAS_MAP[key]
        if key in self.metric_map:     return self.metric_map[key]
        if key in _SEMANTIC_SYNONYMS:  return _SEMANTIC_SYNONYMS[key]

        parts = key.split("_")
        if parts and parts[0].endswith("s"):
            singular = "_".join([parts[0][:-1]] + parts[1:])
            if singular in self.metric_map: return self.metric_map[singular]

        matches = difflib.get_close_matches(raw_metric, self.metric_keys, n=1, cutoff=0.6)
        if matches: return matches[0]

        input_tokens = set(key.split("_")) - _NOISE_TOKENS
        if input_tokens:
            best_key, best_score = None, 0
            for csm_key_lower, csm_key in self.metric_map.items():
                csm_tokens = set(csm_key_lower.split("_")) - _NOISE_TOKENS
                matched = input_tokens & csm_tokens
                if matched == input_tokens and len(matched) > best_score:
                    best_score, best_key = len(matched), csm_key
            if best_key: return best_key

        return raw_metric

    # -----------------------------------------------------------------------
    # Step 3 — Dimension inference
    # -----------------------------------------------------------------------

    def _infer_dimensions(self, intent: dict, q_lower: str) -> dict:
        existing = set(intent.get("dimensions", []))
        inferred = []
        for keyword, dim_keys in _SUBJECT_DIM_HINTS.items():
            if re.search(rf"\b{keyword}\b", q_lower):
                for dk in dim_keys:
                    if dk in self.dimension_map.values() and dk not in existing:
                        inferred.append(dk)
                        existing.add(dk)
        intent["dimensions"] = list(intent.get("dimensions", [])) + inferred
        return intent

    # -----------------------------------------------------------------------
    # Step 4 — RAG hint application
    # -----------------------------------------------------------------------

    def _apply_safe_rag_hints(self, intent: dict, q_lower: str, rag_hints: dict) -> dict:
        if not rag_hints.get("retrieved"):
            return intent
        pattern_intent = rag_hints.get("pattern_intent")
        if not pattern_intent:
            return intent

        # Reject RAG "rate" metric if user clearly wants volume
        if "rate" in pattern_intent.get("metric", "").lower():
            if "most" in q_lower and "efficient" not in q_lower and "rate" not in q_lower:
                return intent

        # Apply metric if it resolved cleanly
        pattern_metric = pattern_intent.get("metric", "")
        if pattern_metric and pattern_metric in self.metric_map.values():
            intent["metric"] = pattern_metric

        # If user asked by category only, strip film-level dimensions from pattern
        category_only = any(w in q_lower for w in ["category", "genre"]) and \
                        not any(w in q_lower for w in ["film", "title", "movie", "per film"])
        if category_only:
            pattern_intent["dimensions"] = [
                d for d in pattern_intent.get("dimensions", [])
                if self.csm.get("dimensions", {}).get(d, {}).get("source") not in ("film", "film_actor", "film_text")
            ]

        # Apply dimensions only if LLM left them empty
        rag_dims = pattern_intent.get("dimensions") or []
        if rag_dims and not intent.get("dimensions"):
            metric_node = self.csm.get('metrics', {}).get(intent.get('metric', ''), {})
            if metric_node.get('join_path'):
                print("  [IntentParser] Metric has explicit join_path — skipping RAG dimension override.")
                return intent
            intent["dimensions"] = rag_dims

        # Copy partition_by from matched pattern — never a name column
        partition_by = [
            d for d in (pattern_intent.get("partition_by") or [])
            if d not in _NAME_DIMS
        ]
        if partition_by:
            intent["partition_by"] = partition_by
            print(f"  [IntentParser] partition_by from RAG pattern: {partition_by}")

        # Apply mode/sort/limit from pattern if high-confidence
        pattern_score = rag_hints.get("pattern_score", 0.0)
        if pattern_score >= 0.72:
            if pattern_intent.get("mode"):
                intent["mode"] = pattern_intent["mode"]
            if pattern_intent.get("sort") is not None:
                intent["sort"] = pattern_intent["sort"]
            if pattern_intent.get("limit") is not None:
                intent["limit"] = pattern_intent["limit"]

        return intent

    # -----------------------------------------------------------------------
    # Step 5 — Dimension validation
    # -----------------------------------------------------------------------

    def _validate_dimensions(self, intent: dict) -> dict:
        valid = []
        for d in intent.get("dimensions", []):
            if d in self.dimension_map.values():
                valid.append(d)
            elif d.lower() in self.dimension_map:
                valid.append(self.dimension_map[d.lower()])
        intent["dimensions"] = valid
        return intent

    def _filter_irrelevant_dimensions(self, intent: dict) -> dict:
        metric_key = intent.get("metric", "")
        metric_node = self.csm.get("metrics", {}).get(metric_key, {})
        allowed_tables = set(metric_node.get("sources", []) + metric_node.get("join_path", []))
        
        if not allowed_tables:
            return intent  # can't validate, leave as-is
        
        filtered = []
        for d in intent.get("dimensions", []):
            dim_node = self.csm.get("dimensions", {}).get(d, {})
            if dim_node.get("source") in allowed_tables:
                filtered.append(d)
            else:
                print(f"  [IntentParser] Dropped dimension '{d}' — source table not in metric's join path.")
        
        intent["dimensions"] = filtered
        return intent

    # -----------------------------------------------------------------------
    # Step 6 — Status filter injection
    # -----------------------------------------------------------------------

    def _inject_status_filter(self, intent: dict, q_lower: str) -> dict:
        for pattern, (col_key, val) in _STATUS_MAP.items():
            if re.search(pattern, q_lower):
                existing = [f for f in intent.get("filters", [])
                            if f.get("col_key") != col_key]
                intent["filters"] = existing + [{
                    "col_key": col_key, "val": val,
                    "op": "equals", "is_aggregate": False
                }]
                break
        return intent