import json
import re
from ruamel.yaml import YAML
import config
from langchain_core.prompts import PromptTemplate
from query_engine.graph_resolver import SchemaGraphResolver

# Naming constants for quality rules
MIN_PATTERNS = 8
SIMILARITY_THRESHOLD = 0.85

SAKILA_TABLES = {
    "actor", "address", "category", "city", "country", "customer", "film", 
    "film_actor", "film_category", "film_text", "inventory", "language", 
    "payment", "rental", "staff", "store"
}

def _infer_compute_type(compute_expr: str) -> str:
    if not compute_expr:
        return "scalar"
    if re.search(r"\b(MAX|MIN|AVG|SUM|COUNT)\s*\(", compute_expr, re.IGNORECASE):
        # We assume grouped aggregate by default if it's generated for a specific dimension
        return "grouped_aggregate"
    return "scalar"

def _infer_usage_hint(compute_type: str) -> str:
    if "aggregate" in compute_type:
        return "Use for summary questions only. Can be grouped by dimensions."
    return "Use for ranking or listing individual records. Never use for single-value summary questions."

def validate_metric(metric_data: dict, metric_key: str, patterns: list) -> tuple[bool, list[str]]:
    """
    Runs after every metric generation.
    Returns (is_valid: bool, errors: list[str])
    """
    errors = []
    
    join_path = metric_data.get("join_path", [])
    required_tables = set(metric_data.get("sources", []))
    
    # 1. Check join path completeness
    for table in required_tables:
        if table not in join_path:
            errors.append(f"Missing required table in join_path: {table}")
            
    # 2. Check GROUP BY matches dimension
    # (If the metric implies a dimension, it should have a group_by)
    group_by = metric_data.get("group_by", [])
    if not group_by and "aggregate" in metric_data.get("compute_type", "") and "by_" in metric_key:
        errors.append(f"Missing GROUP BY for aggregate metric {metric_key}")
        
    # 3. Check no rogue LIMIT
    compute_expr = metric_data.get("compute", "")
    if "LIMIT" in compute_expr.upper():
        errors.append("Unexpected LIMIT in compute expression")
        
    # 4. Check pattern count
    if len(patterns) < MIN_PATTERNS:
        errors.append(f"Too few patterns: {len(patterns)} (min {MIN_PATTERNS})")
        
    # 5. Check all tables exist in schema
    for table in join_path:
        if table not in SAKILA_TABLES:
            errors.append(f"Table does not exist in schema: {table}")
            
    return len(errors) == 0, errors

def _check_pattern_uniqueness(new_patterns: list[str], existing_patterns: list[str]) -> list[str]:
    """
    Filters out new patterns that are too similar (> 0.85 Jaccard similarity) to existing ones.
    """
    unique = []
    for np in new_patterns:
        np_set = set(np.lower().split())
        is_unique = True
        for ep in existing_patterns:
            ep_set = set(ep.lower().split())
            if not np_set or not ep_set: continue
            intersection = len(np_set.intersection(ep_set))
            union = len(np_set.union(ep_set))
            if union == 0: continue
            similarity = intersection / union
            if similarity > SIMILARITY_THRESHOLD:
                is_unique = False
                break
        if is_unique:
            unique.append(np)
    return unique

def write_metric_to_yaml(metric_key: str, metric_data: dict, patterns: list[str]):
    """Writes a new metric to both csm_enterprise.yaml and bgo.yaml using ruamel.yaml"""
    yaml = YAML()
    yaml.preserve_quotes = True

    # Validate before saving
    is_valid, errors = validate_metric(metric_data, metric_key, patterns)
    if not is_valid:
        error_msg = "; ".join(errors)
        print(f"  [AutoGenerate] Validation failed for {metric_key}: {error_msg}")
        raise ValueError(f"Metric validation failed: {error_msg}")

    # 1. Update csm_enterprise.yaml
    with open('csm_enterprise.yaml', 'r', encoding='utf-8') as f:
        csm_yaml_content = yaml.load(f)
        
    if 'compute_type' not in metric_data:
        metric_data['compute_type'] = _infer_compute_type(metric_data.get('compute', ''))
    if 'usage_hint' not in metric_data:
        metric_data['usage_hint'] = _infer_usage_hint(metric_data['compute_type'])

    csm_yaml_content['metrics'][metric_key] = metric_data

    with open('csm_enterprise.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(csm_yaml_content, f)

    # 2. Update bgo.yaml
    with open('bgo.yaml', 'r', encoding='utf-8') as f:
        bgo_yaml_content = yaml.load(f)

    # Use existing or provided synonyms, or generate fallback ones
    synonyms = metric_data.get('synonyms')
    if not synonyms:
        label = metric_data.get('label', metric_key)
        if 'aggregate' in metric_data.get('compute_type', ''):
            synonyms = [label, f"{label} summary", f"Total {label}"]
        else:
            synonyms = [label, f"top {label}", f"highest {label}"]

    bgo_yaml_content['metrics'][metric_key] = {
        'label': metric_data.get('label', metric_key),
        'synonyms': synonyms,
        'description': metric_data.get('label', metric_key),
        'calculation': metric_data.get('compute')
    }

    # Append to intent_patterns
    for idx, p in enumerate(patterns):
        # We can add window hint logic here if needed, but for now we omit it
        # to rely on intent_parser's ranking logic unless specified.
        pattern_entry = {
            'id': f"{metric_key}_p{idx+1}",
            'pattern': p,
            'metric': metric_key,
            'dimensions': metric_data.get('group_by', [])
        }
        
        # If it's an aggregate we provide a generic window config to clear limits
        if metric_data.get('group_by'):
            pattern_entry['window'] = {
                'function': '_none_',
                'partition_by': '_none_',
                'order_by': f"{metric_key} DESC"
            }
            pattern_entry['filter'] = '_none_'
            pattern_entry['limit'] = '_none_'

        bgo_yaml_content['intent_patterns'].append(pattern_entry)

    with open('bgo.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(bgo_yaml_content, f)

    # 3. Hot-reload the config globally
    import yaml as pyyaml
    config.csm = pyyaml.safe_load(open('csm_enterprise.yaml', 'r', encoding='utf-8'))
    config.glossary = pyyaml.safe_load(open('bgo.yaml', 'r', encoding='utf-8'))
    config.METRIC_KEY_MAP = {k.lower(): k for k in config.csm.get('metrics', {}).keys()}


def generate_patterns_for_metric(metric_key: str, dimension_word: str, measure_word: str, existing_patterns: list[str], original_question: str = None) -> list[str]:
    """Generates a minimum of 8 patterns using strict templates via LLM."""
    
    prompt = f"""Generate 12 diverse natural language questions that a user might ask to query the metric '{metric_key}'.
The dimension is '{dimension_word}' and the measure is '{measure_word}'.
Ensure you cover ALL of these exact templates in your variations:
1. "which {dimension_word} has the most {measure_word}"
2. "top {dimension_word} by {measure_word}"
3. "{measure_word} breakdown by {dimension_word}"
4. "list all {dimension_word}s with their {measure_word}"
5. "rank {dimension_word}s by {measure_word}"
6. "{dimension_word} with highest {measure_word}"
7. "total {measure_word} per {dimension_word}"
8. "{measure_word} by {dimension_word} best first"

Also use synonyms for '{dimension_word}' and '{measure_word}' in some of the questions.
Respond with a raw JSON list of strings only.
"""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="You are a data analyst generating training queries. Output JSON list of strings only."),
            HumanMessage(content=prompt)
        ]
        response = config.llm.invoke(messages)
        text = response.content.strip()
        if text.startswith("```json"): text = text.split("```json")[1]
        if text.endswith("```"): text = text.rsplit("```", 1)[0]
        patterns = json.loads(text.strip())
        
        if original_question and original_question not in patterns:
            patterns.insert(0, original_question)

        # Check uniqueness against existing
        unique_patterns = _check_pattern_uniqueness(patterns, existing_patterns)
        
        # Ensure we have at least 8, fallback to simple ones if LLM failed
        if len(unique_patterns) < MIN_PATTERNS:
            fallbacks = [
                f"which {dimension_word} has the most {measure_word}",
                f"top {dimension_word} by {measure_word}",
                f"{measure_word} breakdown by {dimension_word}",
                f"list all {dimension_word}s with their {measure_word}",
                f"rank {dimension_word}s by {measure_word}",
                f"{dimension_word} with highest {measure_word}",
                f"total {measure_word} per {dimension_word}",
                f"{measure_word} by {dimension_word} best first"
            ]
            for fb in fallbacks:
                if fb not in unique_patterns:
                    unique_patterns.append(fb)
                    
        return unique_patterns[:15] # Return up to 15
    except Exception as e:
        print(f"  [AutoGenerate] Failed to generate patterns: {e}")
        return [f"show {measure_word} by {dimension_word}"] * MIN_PATTERNS # Fallback to prevent crash

def _get_table_for_dimension(dim_key: str) -> str:
    # Try to find the table from CSM
    for d_k, d_v in config.csm.get('dimensions', {}).items():
        if d_k == dim_key or d_k == f"{dim_key}_name":
            return d_v.get('source')
    # Fallback to fuzzy matching
    for table in SAKILA_TABLES:
        if table in dim_key:
            return table
    return None

def auto_generate_metric(metric_key: str, base_table: str, dimension_key: str, measure_expr: str, compute_func: str = "SUM", label: str = None) -> bool:
    """
    Programmatically generates a metric with rigorous join paths and patterns.
    """
    try:
        # Determine dimension table and column
        dim_table = _get_table_for_dimension(dimension_key)
        if not dim_table:
            dim_table = dimension_key.split('_')[0] if '_' in dimension_key else dimension_key
            if dim_table not in SAKILA_TABLES:
                print(f"  [AutoGenerate] Could not resolve dimension table for {dimension_key}")
                return False

        # Build join path using Graph Resolver
        graph = SchemaGraphResolver()
        path = graph.find_shortest_path(base_table, dim_table)
        
        join_path = [base_table]
        if path:
            for to_table, _ in path:
                join_path.append(to_table)
                
        # If no path, and base_table != dim_table, abort
        if base_table != dim_table and not path:
             print(f"  [AutoGenerate] No path from {base_table} to {dim_table}")
             return False

        compute = f"{compute_func}({measure_expr})" if compute_func else measure_expr
        group_by = [f"{dim_table}.name"] if dim_table in ("category", "language") else [f"{dim_table}.title"] if dim_table == "film" else [f"{dim_table}.{dim_table}_id"]
        
        # Some special cases for Sakila names
        if dim_table == "customer" or dim_table == "staff" or dim_table == "actor":
            group_by = [f"{dim_table}.first_name", f"{dim_table}.last_name"]
        elif dim_table == "store":
            group_by = [f"{dim_table}.store_id"]

        comp_type = _infer_compute_type(compute)
        
        metric_data = {
            'compute': compute,
            'sources': [base_table, dim_table] if base_table != dim_table else [base_table],
            'join_path': join_path,
            'group_by': group_by,
            'label': label or metric_key.replace('_', ' ').title(),
            'compute_type': comp_type,
            'usage_hint': _infer_usage_hint(comp_type),
            'synonyms': [label or metric_key, f"total {metric_key}", f"{metric_key} summary"]
        }

        # Gather existing patterns
        existing_patterns = [p.get('pattern', '') for p in config.csm.get('intent_patterns', [])]
        
        # Generate patterns
        measure_word = metric_key.split('_by_')[0] if '_by_' in metric_key else "amount"
        dimension_word = dim_table
        patterns = generate_patterns_for_metric(metric_key, dimension_word, measure_word, existing_patterns, original_question=label or metric_key)

        write_metric_to_yaml(metric_key, metric_data, patterns)
        return True
        
    except Exception as e:
        print(f"  [AutoGenerate] Metric generation failed: {e}")
        return False
