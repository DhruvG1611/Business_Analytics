"""
patch_yaml.py  (v2)
-------------------
Run once from your project directory:
    python patch_yaml.py

Adds project-level skill metrics to csm_enterprise.yaml with explicit
join_path hints so the BFS resolver always takes the correct path:
    employee_skills -> employees -> tasks -> projects
instead of the wrong shortcut:
    employee_skills -> employees -> departments -> projects

Also patches bgo.yaml with synonyms and intent patterns.
"""
import yaml
import sys
from pathlib import Path

CSM_PATH = Path("csm_enterprise.yaml")
BGO_PATH = Path("bgo.yaml")

for p in (CSM_PATH, BGO_PATH):
    if not p.exists():
        print(f"ERROR: {p} not found. Run from your project directory.")
        sys.exit(1)


# ===========================================================================
# 1. Patch csm_enterprise.yaml
# ===========================================================================

with open(CSM_PATH) as f:
    csm = yaml.safe_load(f)

NEW_METRICS = {
    'total_proficiency_per_project': {
        'compute': 'SUM(employee_skills.proficiency)',
        'sources': ['employee_skills', 'employees', 'tasks', 'projects'],
        'join_path': ['employee_skills', 'employees', 'tasks', 'projects'],
        'label': 'Total Skill Proficiency per Project',
        'description': (
            'Sum of proficiency scores of all employees assigned to tasks '
            'within a project. Best proxy for most skilled people working on a project.'
        ),
    },
    'avg_proficiency_per_project': {
        'compute': 'AVG(employee_skills.proficiency)',
        'sources': ['employee_skills', 'employees', 'tasks', 'projects'],
        'join_path': ['employee_skills', 'employees', 'tasks', 'projects'],
        'label': 'Average Skill Proficiency per Project',
    },
    'distinct_skills_per_project': {
        'compute': 'COUNT(DISTINCT employee_skills.skill_id)',
        'sources': ['employee_skills', 'employees', 'tasks', 'projects'],
        'join_path': ['employee_skills', 'employees', 'tasks', 'projects'],
        'label': 'Distinct Skills per Project',
    },
    'skilled_employees_per_project': {
        'compute': 'COUNT(DISTINCT employees.id)',
        'sources': ['employee_skills', 'employees', 'tasks', 'projects'],
        'join_path': ['employee_skills', 'employees', 'tasks', 'projects'],
        'label': 'Skilled Employees per Project',
    },
    # Fix existing department skill metrics with correct path hints too
    'avg_proficiency_per_department': {
        'compute': 'AVG(employee_skills.proficiency)',
        'sources': ['employee_skills', 'employees', 'departments'],
        'join_path': ['employee_skills', 'employees', 'departments'],
        'label': 'Average Proficiency per Department',
    },
    'skill_diversity_per_department': {
        'compute': 'COUNT(DISTINCT employee_skills.skill_id)',
        'sources': ['employee_skills', 'employees', 'departments'],
        'join_path': ['employee_skills', 'employees', 'departments'],
        'label': 'Skill Diversity per Department',
    },
    'employees_with_skill_per_department': {
        'compute': 'COUNT(DISTINCT employees.id)',
        'sources': ['employee_skills', 'employees', 'departments'],
        'join_path': ['employee_skills', 'employees', 'departments'],
        'label': 'Skilled Employees per Department',
    },
}

added_m, updated_m = [], []
for key, val in NEW_METRICS.items():
    if key not in csm['metrics']:
        csm['metrics'][key] = val
        added_m.append(key)
    else:
        csm['metrics'][key].update(val)
        updated_m.append(key)

with open(CSM_PATH, 'w') as f:
    yaml.dump(csm, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False, indent=2)

print("csm_enterprise.yaml:")
if added_m:   print(f"  Added   : {added_m}")
if updated_m: print(f"  Updated : {updated_m}")


# ===========================================================================
# 2. Patch bgo.yaml
# ===========================================================================

with open(BGO_PATH) as f:
    bgo = yaml.safe_load(f)

bgo.setdefault('metrics', {})
bgo.setdefault('intent_patterns', [])

NEW_BGO_METRICS = {
    'total_proficiency_per_project': [
        'most skilled project team',
        'which project has the most skilled people',
        'which project has the most skilled people working on it',
        'project with highest skill',
        'most talented project team',
        'best skilled project',
        'highest skill score per project',
        'total skill per project',
        'project skill total',
        'which project has the best team',
        'most capable project team',
        'project skill strength',
        'skill power per project',
        'project with most talent',
        'highest combined skill project',
    ],
    'avg_proficiency_per_project': [
        'average skill per project',
        'mean skill level per project',
        'project skill average',
        'most skilled project on average',
        'highest average skill project',
        'project proficiency average',
    ],
    'distinct_skills_per_project': [
        'skill diversity per project',
        'distinct skills per project',
        'how many skills per project',
        'skill variety in project',
        'project skill breadth',
        'most diverse skill set per project',
    ],
    'skilled_employees_per_project': [
        'skilled employees per project',
        'how many skilled people per project',
        'employees with skills per project',
    ],
}

for key, synonyms in NEW_BGO_METRICS.items():
    if key not in bgo['metrics']:
        bgo['metrics'][key] = synonyms
    else:
        existing = set(bgo['metrics'][key])
        bgo['metrics'][key].extend(s for s in synonyms if s not in existing)

NEW_PATTERNS = [
    {
        'pattern': 'Which project has the most skilled people working on it?',
        'metric': 'total_proficiency_per_project',
        'dimensions': ['projects_project_name'],
        'sort': 'desc',
        'limit': 1,
    },
    {
        'pattern': 'Which project has the most skilled team?',
        'metric': 'total_proficiency_per_project',
        'dimensions': ['projects_project_name'],
        'sort': 'desc',
        'limit': 1,
    },
    {
        'pattern': 'Rank projects by team skill level',
        'metric': 'total_proficiency_per_project',
        'dimensions': ['projects_project_name'],
        'sort': 'desc',
    },
    {
        'pattern': 'Which project has the highest average skill proficiency?',
        'metric': 'avg_proficiency_per_project',
        'dimensions': ['projects_project_name'],
        'sort': 'desc',
        'limit': 1,
    },
    {
        'pattern': 'Which project has the most skill diversity?',
        'metric': 'distinct_skills_per_project',
        'dimensions': ['projects_project_name'],
        'sort': 'desc',
        'limit': 1,
    },
]

existing_patterns = {p.get('pattern', '') for p in bgo['intent_patterns']}
added_p = []
for p in NEW_PATTERNS:
    if p['pattern'] not in existing_patterns:
        bgo['intent_patterns'].append(p)
        added_p.append(p['pattern'])

with open(BGO_PATH, 'w') as f:
    yaml.dump(bgo, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False, indent=2)

print(f"bgo.yaml: {len(NEW_BGO_METRICS)} metric synonym blocks, {len(added_p)} new patterns")
print()
print("Done. Restart connector.py.")
print()
print("Expected SQL:")
print("  SELECT projects.project_name, SUM(employee_skills.proficiency) AS result")
print("  FROM employee_skills")
print("  LEFT JOIN employees ON employee_skills.employee_id = employees.id")
print("  LEFT JOIN tasks ON tasks.assigned_to = employees.id")
print("  LEFT JOIN projects ON tasks.project_id = projects.id")
print("  GROUP BY projects.project_name")
print("  ORDER BY result DESC")
print("  LIMIT 1")