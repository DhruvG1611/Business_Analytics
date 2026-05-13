import sys
import os
sys.path.append(os.getcwd())

import yaml
import config
from core.intent_parser import IntentParser
from query_engine.graph_resolver import rag_plus_plus_resolver
from query_engine.sql_compiler import sql_compiler

# Load CSM
with open("csm_enterprise.yaml", encoding="utf-8") as f:
    config.csm = yaml.safe_load(f)

# Load Glossary (BGO)
with open("bgo.yaml", encoding="utf-8") as f:
    glossary = yaml.safe_load(f)

parser = IntentParser(config.csm, glossary)

def test_query(question):
    print(f"\n--- Testing: {question} ---")
    
    # 1. Intent Parsing
    raw_intent = {"metric": "copies_per_film"}
    rag_hints = {"retrieved": True, "pattern_intent": {"mode": "set_difference", "metric": "copies_per_film", "dimensions": ["film_film_id", "film_title"]}}
    
    intent_res = parser.process_intent(raw_intent, question, rag_hints)
    intent = intent_res["intent"]
    print(f"Detected Mode: {intent.get('mode')}")
    print(f"Include Store: {intent.get('_include_store')}")
    print(f"Exclude Store: {intent.get('_exclude_store')}")
    
    # 2. Graph Resolution
    plan = rag_plus_plus_resolver(intent_res)
    print(f"Plan Mode: {plan.get('mode')}")
    print(f"Exclude Subquery: {plan.get('exclude_subquery')}")
    
    # 3. SQL Compilation
    sql, params = sql_compiler(plan)
    print("Generated SQL:")
    print(sql)
    print(f"Params: {params}")

# Test 1: Full inclusion/exclusion
test_query("Find the films that are in inventory at store 1 but NOT at store 2")

# Test 2: Pure exclusion
test_query("Titles in stock excluding store 2")

# Test 3: Swap stores
test_query("Films at store 2 but not store 1")
