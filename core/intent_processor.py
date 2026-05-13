"""intent_processor.py — Input validation and question preprocessing.

normalize_intent() was removed: it was dead code duplicating logic already
in IntentParser._detect_mode (which is the active code path in pipeline.py).
The HAVING patterns it used now live in core/patterns.py.
"""
import re
from config import METRIC_KEY_MAP, DIMENSION_KEY_MAP

class NotAQuestionError(ValueError): pass

def validate_question(question: str) -> str:
    q = question.strip()
    if len(q) < 3: raise NotAQuestionError("Input is too short.")
    return q