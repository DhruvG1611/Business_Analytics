import json
from ruamel.yaml import YAML
import config
from langchain_core.prompts import PromptTemplate

def write_metric_to_yaml(metric_key: str, metric_data: dict, question: str):
    """Writes a new metric to both csm_enterprise.yaml and bgo.yaml using ruamel.yaml"""
    yaml = YAML()
    yaml.preserve_quotes = True

    # 1. Update csm_enterprise.yaml
    with open('csm_enterprise.yaml', 'r', encoding='utf-8') as f:
        csm_yaml_content = yaml.load(f)
        
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
        synonyms = [label, f"{label} count", f"Total {label}"]

    bgo_yaml_content['metrics'][metric_key] = {
        'label': metric_data.get('label', metric_key),
        'synonyms': synonyms,
        'description': metric_data.get('label', metric_key),
        'calculation': metric_data.get('compute')
    }

    # Append to intent_patterns
    bgo_yaml_content['intent_patterns'].append({
        'pattern': question,
        'metric': metric_key,
        'dimensions': []
    })
    bgo_yaml_content['intent_patterns'].append({
        'pattern': metric_data.get('label', metric_key),
        'metric': metric_key,
        'dimensions': []
    })

    with open('bgo.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(bgo_yaml_content, f)

    # 3. Hot-reload the config globally
    import yaml as pyyaml
    config.csm = pyyaml.safe_load(open('csm_enterprise.yaml', 'r', encoding='utf-8'))
    config.glossary = pyyaml.safe_load(open('bgo.yaml', 'r', encoding='utf-8'))
    config.METRIC_KEY_MAP = {k.lower(): k for k in config.csm.get('metrics', {}).keys()}


def auto_generate_metric(metric_key: str, question: str, intent: dict) -> bool:
    """Attempts to generate a missing metric using the LLM."""
    
    dimensions = config.csm.get('dimensions', {})
    relationships = config.csm.get('relationships', [])
    
    # Build a compact representation of tables and columns
    tables = {}
    for d_key, d_val in dimensions.items():
        src = d_val.get('source')
        if src not in tables:
            tables[src] = []
        tables[src].append(d_val.get('column'))

    tables_str = "\n".join([f"- {tbl}: {', '.join(cols)}" for tbl, cols in tables.items()])
    rels_str = "\n".join([f"- {r}" for r in relationships])

    system_prompt = f"""You are an expert data architect. Generate a missing metric definition for a Semantic Layer.

AVAILABLE TABLES AND COLUMNS:
{tables_str}

EXISTING RELATIONSHIPS:
{rels_str}

EXAMPLE VALID METRIC ENTRY (rentals_per_film):
{{
  "compute": "COUNT(DISTINCT rental.rental_id)",
  "sources": ["rental", "inventory", "film"],
  "label": "Rentals per Film",
  "join_path": ["rental", "inventory", "film"]
}}

INSTRUCTIONS:
- Respond ONLY in raw JSON, no markdown formatting, no explanations.
- Do not wrap the JSON in ```json blocks.
- Generate synonyms (list of 3 strings) based on the label.
"""

    user_prompt = f"""The user asked: "{question}"
The resolved metric key is: "{metric_key}"
Generate a valid CSM metric definition for this metric key.
If the compute expression requires a CTE (e.g. cohort comparisons, month-over-month, churn), also include "is_cte_metric": true and "cte_definition": "<the full WITH clause body, without the WITH keyword>".
Respond with a single JSON object with keys: compute, sources (list), label, join_path (list, optional), is_cte_metric (bool, optional), cte_definition (string, optional), synonyms (list)."""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Parse JSON
        response_text = response.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()
            
        parsed = json.loads(response_text)
        
        if 'compute' not in parsed or 'sources' not in parsed:
            return False
            
        # Clean up output format
        metric_data = {
            'compute': parsed['compute'],
            'sources': parsed['sources'],
            'label': parsed.get('label', metric_key)
        }
        
        if 'join_path' in parsed:
            metric_data['join_path'] = parsed['join_path']
        if parsed.get('is_cte_metric'):
            metric_data['is_cte_metric'] = True
        if 'cte_definition' in parsed:
            metric_data['cte_definition'] = parsed['cte_definition']
        if 'synonyms' in parsed:
            metric_data['synonyms'] = parsed['synonyms']
            
        write_metric_to_yaml(metric_key, metric_data, question)
        return True
        
    except Exception as e:
        print(f"  [AutoGenerate] Failed to parse or generate metric: {e}")
        return False
