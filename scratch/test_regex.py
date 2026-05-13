import re
q = "find the films that are in inventory at store 1 but not at store 2"

patterns = [
    r"\bbut\s+not\s+(in|at)\b",
    r"\bnot\s+(in|at)\s+store\b",
    r"\bonly\s+(in|at)\s+store\b",
    r"\bin\s+store\s*\d+\s+but\s+not\b",
    r"\bnot\s+available\s+(in|at)\b",
]
for p in patterns:
    m = re.search(p, q)
    print(f"{p!r:50s} -> {'MATCH' if m else 'no match'}")
