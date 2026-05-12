import os
from dotenv import load_dotenv
load_dotenv()
import yaml
import mysql.connector
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load Models
csm = yaml.safe_load(open('csm_enterprise.yaml', 'r', encoding='utf-8'))
glossary = yaml.safe_load(open('bgo.yaml', 'r', encoding='utf-8'))

# Setup Connections
def get_db_connection():
    """Create a connection to the Sakila database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sakila",
        port=3306
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load Prompt
with open("decomposition_prompt.txt", encoding="utf-8") as _f:
    _PROMPT_TEMPLATE = _f.read()

decomposition_chain = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE) | llm | JsonOutputParser()

# Pre-compute lookup maps
METRIC_KEY_MAP = {k.lower(): k for k in csm.get('metrics', {}).keys()}
DIMENSION_KEY_MAP = {k.lower(): k for k in csm.get('dimensions', {}).keys()}

# PII columns — cross-reference bgo.yaml (pii flag) with csm (source/column).
# Glossary dimensions have pii: true but no source/column;
# CSM dimensions have source/column but no pii flag.
# We match by dimension key across both dicts.
def _build_pii_columns() -> frozenset:
    pii_keys = {
        k for k, v in glossary.get("dimensions", {}).items()
        if v.get("pii") is True
    }
    result = set()
    for k in pii_keys:
        csm_dim = csm.get("dimensions", {}).get(k, {})
        source = csm_dim.get("source", "")
        column = csm_dim.get("column", "")
        if source and column:
            result.add(f"{source}.{column}")
    return frozenset(result)

PII_COLUMNS: frozenset = _build_pii_columns()