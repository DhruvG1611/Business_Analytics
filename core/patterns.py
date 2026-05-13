"""
patterns.py
-----------
Shared regex patterns used by both intent_parser.py and intent_processor.py.
Consolidated here to avoid duplication — single source of truth.
"""

HAVING_PATTERNS = [
    (r"at least\s+(\d+)",   "gte"),
    (r"more than\s+(\d+)",  "gt"),
    (r"over\s+(\d+)",       "gt"),
    (r"at most\s+(\d+)",    "lte"),
    (r"fewer than\s+(\d+)", "lt"),
    (r"less than\s+(\d+)",  "lt"),
]

SET_DIFFERENCE_PATTERNS = [
    r"\bbut\s+not\b",
    r"\bexcept\b",
    r"\bnot\sin\b",
    r"\bexcluding\b",
]
