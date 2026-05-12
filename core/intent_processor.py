import re
from config import METRIC_KEY_MAP, DIMENSION_KEY_MAP

class NotAQuestionError(ValueError): pass

def validate_question(question: str) -> str:
    q = question.strip()
    if len(q) < 3: raise NotAQuestionError("Input is too short.")
    # (Include your existing _SHELL_PATTERNS check here)
    return q

def normalize_intent(intent_output: dict, question: str) -> dict:
    data = intent_output.get('intent', intent_output)
    
    # 1. Normalize Keys
    raw_metric = data.get("metric", "")
    data["metric"] = METRIC_KEY_MAP.get(raw_metric.lower(), raw_metric)
    data["dimensions"] = [DIMENSION_KEY_MAP.get(d.lower(), d) for d in data.get("dimensions", []) if d.lower() in DIMENSION_KEY_MAP]
    
    # 2. Extract HAVING Thresholds accurately
    _HAVING_PATTERNS = [
        (r'at least\s+(\d+)', 'gte'), (r'more than\s+(\d+)', 'gt'),
        (r'at most\s+(\d+)', 'lte'), (r'fewer than\s+(\d+)', 'lt'),
    ]
    q_lower = question.lower()
    
    if not data.get('having_threshold'):
        for pattern, op in _HAVING_PATTERNS:
            m = re.search(pattern, q_lower)
            if m:
                data['having_threshold'] = {'op': op, 'val': int(m.group(1))}
                # Clear invalid sorts mapped by ranking enforcer
                data['sort'], data['limit'] = None, None
                break
                
    return data

# (Include your existing enforce_ranking, enforce_status_filter, and _apply_rag_hints here seamlessly)