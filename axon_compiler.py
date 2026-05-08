"""
axon_compiler.py
----------------
Compiles AXON DSL (axon.yaml) plan nodes into executable SQL.

Architecture:
    AxonLoader      → parses + validates axon.yaml, resolves cross-references
    IntentResolver  → maps LLM output keys → measure / dimension / recipe nodes
    QueryPlanner    → decides plan type: simple | aggregate | ratio | window |
                      composite | subquery | filtered_dim | recipe
    SQLBuilder      → dispatches to per-type builder; pure functions
        ├── SimpleBuilder
        ├── AggregateBuilder
        ├── RatioBuilder
        ├── WindowBuilder
        ├── CompositeBuilder
        ├── SubqueryBuilder
        └── RecipeBuilder

Public API:
    compiler = AxonCompiler("axon.yaml")
    sql = compiler.compile(intent_dict)
    # intent_dict keys: metric, dimensions, filters, sort, limit, mode, recipe
"""

from __future__ import annotations

import re
import yaml
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Plan:
    plan_type: str                   # simple | aggregate | ratio | window |
                                     # composite | subquery | filtered_dim | recipe
    measure_key: str = ""
    measure_node: dict = field(default_factory=dict)
    dimensions: list[tuple[str, dict]] = field(default_factory=list)  # [(key, node)]
    filters: list[dict] = field(default_factory=list)
    sort: str | None = None          # asc | desc
    limit: int | None = None
    mode: str | None = None          # list | None
    recipe_key: str = ""
    recipe_node: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# AXON LOADER
# ═══════════════════════════════════════════════════════════════════════════

class AxonLoader:
    """Parses axon.yaml and exposes resolved node lookups."""

    def __init__(self, path: str = "axon.yaml"):
        with open(path, "r") as f:
            self._raw = yaml.safe_load(f)

        self.sources       = self._raw.get("sources", {})
        self.relationships = self._raw.get("relationships", {})
        self.dimensions    = self._raw.get("dimensions", {})
        self.measures      = self._raw.get("measures", {})
        self.recipes       = self._raw.get("recipes", {})

        # Build synonym → key indices (lowercase for case-insensitive lookup)
        self._measure_idx   = self._build_synonym_index(self.measures)
        self._dimension_idx = self._build_synonym_index(self.dimensions)
        self._recipe_idx    = self._build_recipe_synonym_index()

        # Case-insensitive key maps (for exact-key lookups from LLM output)
        self._measure_key_map   = {k.lower(): k for k in self.measures}
        self._dimension_key_map = {k.lower(): k for k in self.dimensions}
        self._recipe_key_map    = {k.lower(): k for k in self.recipes}

    def _build_synonym_index(self, nodes: dict) -> dict[str, str]:
        idx: dict[str, str] = {}
        for key, node in nodes.items():
            idx[key.lower()] = key
            for syn in node.get("synonyms", []):
                idx[syn.lower()] = key
        return idx

    def _build_recipe_synonym_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for key, node in self.recipes.items():
            idx[key.lower()] = key
            for syn in node.get("synonyms", []):
                idx[syn.lower()] = key
        return idx

    # ── Resolvers ────────────────────────────────────────────────────────

    def resolve_measure(self, raw: str) -> tuple[str, dict] | tuple[None, None]:
        canonical = self._measure_key_map.get(raw.lower()) or self._measure_idx.get(raw.lower())
        if canonical:
            return canonical, self.measures[canonical]
        return None, None

    def resolve_dimension(self, raw: str) -> tuple[str, dict] | tuple[None, None]:
        canonical = self._dimension_key_map.get(raw.lower()) or self._dimension_idx.get(raw.lower())
        if canonical:
            return canonical, self.dimensions[canonical]
        return None, None

    def resolve_recipe(self, raw: str) -> tuple[str, dict] | tuple[None, None]:
        canonical = self._recipe_key_map.get(raw.lower()) or self._recipe_idx.get(raw.lower())
        if canonical:
            return canonical, self.recipes[canonical]
        return None, None

    # ── Graph traversal: join path BFS ───────────────────────────────────

    def join_path(self, start: str, target: str) -> list[tuple[str, str, str]] | None:
        """
        Returns list of (to_table, join_type, ON_clause) edges from start → target.
        Handles aliases for self-joins.
        """
        if start == target:
            return []
        queue:   list[tuple[str, list]] = [(start, [])]
        visited: set[str] = {start}

        while queue:
            current, path = queue.pop(0)
            for rel_key, rel in self.relationships.items():
                alias = rel.get("via_alias")
                jtype = rel.get("join", "LEFT")

                if rel["from"] == current:
                    to = rel["to"]
                    on = rel.get("on") or rel.get(True, "")
                    label = alias if alias else to
                    if label not in visited:
                        new_path = path + [(label, jtype, on)]
                        if label == target or to == target:
                            return new_path
                        visited.add(label)
                        queue.append((to, new_path))

                elif rel["to"] == current and not alias:
                    frm = rel["from"]
                    on  = rel.get("on") or rel.get(True, "")
                    if frm not in visited:
                        new_path = path + [(frm, jtype, on)]
                        if frm == target:
                            return new_path
                        visited.add(frm)
                        queue.append((frm, new_path))
        return None

    def build_joins(self, base_table: str, required_tables: set[str]) -> list[str]:
        """Return ordered JOIN clauses to reach all required_tables from base_table."""
        joined  : set[str] = {base_table}
        clauses : list[str] = []

        for target in required_tables:
            if target in joined:
                continue
            path = self.join_path(base_table, target)
            if path is None:
                print(f"  [warn] No join path from '{base_table}' to '{target}'")
                continue
            for (to_table, jtype, on) in path:
                if to_table not in joined:
                    # Handle self-join aliases
                    rel_with_alias = next(
                        (r for r in self.relationships.values()
                         if r.get("via_alias") == to_table),
                        None,
                    )
                    if rel_with_alias:
                        real_table = rel_with_alias["to"]
                        clauses.append(f"{jtype} JOIN {real_table} AS {to_table} ON {on}")
                    else:
                        clauses.append(f"{jtype} JOIN {to_table} ON {on}")
                    joined.add(to_table)
        return clauses


# ═══════════════════════════════════════════════════════════════════════════
# INTENT RESOLVER
# ═══════════════════════════════════════════════════════════════════════════

class IntentResolver:
    """Maps raw LLM intent dict → validated Plan."""

    def __init__(self, loader: AxonLoader):
        self.axon = loader

    def resolve(self, intent: dict) -> Plan:
        # ── Recipe shortcut ──────────────────────────────────────────────
        raw_recipe = intent.get("recipe", "")
        if raw_recipe:
            rk, rn = self.axon.resolve_recipe(raw_recipe)
            if rk:
                return Plan(plan_type="recipe", recipe_key=rk, recipe_node=rn,
                            sort=intent.get("sort"), limit=intent.get("limit"))

        # ── Measure ──────────────────────────────────────────────────────
        raw_metric = intent.get("metric", "")
        mk, mn = self.axon.resolve_measure(raw_metric)
        if not mk:
            # Try inferring from question via synonym scan (already done upstream)
            raise ValueError(
                f"Measure '{raw_metric}' not found in axon.yaml. "
                f"Available: {list(self.axon.measures.keys())}"
            )

        # ── Dimensions ───────────────────────────────────────────────────
        dim_nodes: list[tuple[str, dict]] = []
        for raw_dim in intent.get("dimensions", []):
            dk, dn = self.axon.resolve_dimension(raw_dim)
            if dk:
                dim_nodes.append((dk, dn))
            else:
                print(f"  [warn] Dimension '{raw_dim}' not found — skipped")

        # ── Filters ──────────────────────────────────────────────────────
        filters: list[dict] = []
        for f in intent.get("filters", []):
            raw_key = f.get("col_key") or f.get("field") or ""
            dk, dn  = self.axon.resolve_dimension(raw_key)
            if not dk:
                print(f"  [warn] Filter '{raw_key}' not found — skipped")
                continue
            values = (f.get("values") or f.get("vals")
                      or ([f.get("val")] if f.get("val") is not None else []))
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = [item for sub in values for item in sub]
            values = [v for v in values if v is not None and str(v).strip() != ""]
            if not values:
                continue
            filters.append({
                "col_key": dk,
                "dim_node": dn,
                "val": values[0],
                "vals": values,
                "op": f.get("operator") or f.get("op") or "equals",
            })

        # ── Plan type decision ────────────────────────────────────────────
        measure_type = mn.get("type", "count")
        mode         = intent.get("mode")

        if mode == "list":
            ptype = "simple"
        elif measure_type == "ratio":
            ptype = "ratio"
        elif measure_type == "window":
            ptype = "window"
        elif measure_type == "composite":
            ptype = "composite"
        elif measure_type == "subquery":
            ptype = "subquery"
        elif measure_type == "filtered_dimension":
            ptype = "filtered_dim"
        else:
            ptype = "aggregate"

        return Plan(
            plan_type    = ptype,
            measure_key  = mk,
            measure_node = mn,
            dimensions   = dim_nodes,
            filters      = filters,
            sort         = intent.get("sort"),
            limit        = intent.get("limit"),
            mode         = mode,
        )


# ═══════════════════════════════════════════════════════════════════════════
# WHERE CLAUSE BUILDER (shared utility)
# ═══════════════════════════════════════════════════════════════════════════

def build_where_clause(filters: list[dict], axon: AxonLoader) -> str:
    """Compile filter list → SQL WHERE fragment (no WHERE keyword)."""
    clauses = []
    for f in filters:
        dn    = f.get("dim_node") or axon.dimensions.get(f.get("col_key", ""), {})
        col   = dn.get("column", "")
        src   = dn.get("source", "")
        dtype = dn.get("type", "string")
        op    = f.get("op", "equals")
        vals  = f.get("vals", [f.get("val")])
        vals  = [v for v in vals if v is not None]
        if not vals or not col:
            continue

        col_ref = f"{src}.{col}" if src else col

        # last_n_days → DATE function
        if op == "last_n_days":
            try:
                n = int(vals[0])
            except (ValueError, TypeError):
                n = 30
            clauses.append(f"{col_ref} >= DATE_SUB(NOW(), INTERVAL {n} DAY)")
            continue

        if dtype in ("id", "number"):
            try:
                num_vals = [int(v) if str(v).isdigit() else float(v) for v in vals]
            except (ValueError, TypeError):
                num_vals = vals
            v0 = num_vals[0]
            if len(num_vals) > 1:
                in_list = ", ".join(str(v) for v in num_vals)
                clauses.append(f"{col_ref} IN ({in_list})")
            elif op in ("gt", ">"):
                clauses.append(f"{col_ref} > {v0}")
            elif op in ("gte", ">="):
                clauses.append(f"{col_ref} >= {v0}")
            elif op in ("lt", "<"):
                clauses.append(f"{col_ref} < {v0}")
            elif op in ("lte", "<="):
                clauses.append(f"{col_ref} <= {v0}")
            elif op == "notEquals":
                clauses.append(f"{col_ref} != {v0}")
            else:
                clauses.append(f"{col_ref} = {v0}")

        elif dtype == "datetime":
            sv = str(vals[0]).replace("'", "''")
            if op in ("gt", ">"):
                clauses.append(f"{col_ref} > '{sv}'")
            elif op in ("gte", ">="):
                clauses.append(f"{col_ref} >= '{sv}'")
            elif op in ("lt", "<"):
                clauses.append(f"{col_ref} < '{sv}'")
            elif op in ("lte", "<="):
                clauses.append(f"{col_ref} <= '{sv}'")
            elif op == "between" and len(vals) >= 2:
                sv2 = str(vals[1]).replace("'", "''")
                clauses.append(f"{col_ref} BETWEEN '{sv}' AND '{sv2}'")
            else:
                clauses.append(f"{col_ref} = '{sv}'")

        else:  # string / enum
            safe = [str(v).replace("'", "''") for v in vals]
            if len(safe) > 1 or op in ("in", "notIn"):
                in_parts = ", ".join(f"LOWER('{v}')" for v in safe)
                neg = "NOT " if op == "notIn" else ""
                clauses.append(f"LOWER({col_ref}) {neg}IN ({in_parts})")
            elif op == "contains":
                clauses.append(f"LOWER({col_ref}) LIKE LOWER('%{safe[0]}%')")
            elif op == "notEquals":
                clauses.append(f"LOWER({col_ref}) != LOWER('{safe[0]}')")
            else:
                clauses.append(f"LOWER({col_ref}) = LOWER('{safe[0]}')")

    return " AND ".join(clauses)


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL BUILDERS  (pure functions: plan + axon → sql string)
# ═══════════════════════════════════════════════════════════════════════════

def _collect_tables(plan: Plan, base_table: str) -> set[str]:
    """Union of tables needed for dimensions + filters."""
    tables: set[str] = set()
    for _, dn in plan.dimensions:
        tables.add(dn["source"])
    for f in plan.filters:
        dn = f.get("dim_node", {})
        if dn.get("source"):
            tables.add(dn["source"])
    tables.discard(base_table)
    return tables


def _measure_compute(mk: str, mn: dict, axon: AxonLoader, alias: str = "result") -> str:
    """Return the SQL expression for a simple count/sum measure."""
    t = mn.get("type", "count")
    src = mn.get("source", "")
    if t == "count":
        flt = mn.get("filter")
        if flt:
            dim_key = flt["field"]
            dn      = axon.dimensions.get(dim_key, {})
            col_ref = f"{dn.get('source', src)}.{dn.get('column', flt['field'])}"
            op      = flt["op"]
            val     = str(flt["val"]).replace("'", "''")
            dtype   = dn.get("type", "string")
            if dtype in ("id", "number"):
                cond = f"{col_ref} = {val}"
            else:
                cond = f"LOWER({col_ref}) = LOWER('{val}')"
            return f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) AS {alias}"
        return f"COUNT({src}.id) AS {alias}"
    return f"COUNT(*) AS {alias}"


# ── SimpleBuilder ──────────────────────────────────────────────────────────

def simple_builder(plan: Plan, axon: AxonLoader) -> str:
    """SELECT col1, col2 FROM … WHERE …  (no aggregation)."""
    base  = plan.measure_node.get("source", "")
    extra = _collect_tables(plan, base)
    joins = axon.build_joins(base, extra)

    if plan.dimensions:
        cols = [f"{dn['source']}.{dn['column']}" for _, dn in plan.dimensions]
    else:
        cols = [f"{base}.*"]

    sql  = f"SELECT {', '.join(cols)}\nFROM {base}"
    if joins:
        sql += "\n" + "\n".join(joins)

    where = build_where_clause(plan.filters, axon)
    if where:
        sql += f"\nWHERE {where}"

    if plan.sort:
        sql += f"\nORDER BY {cols[-1]} {'DESC' if plan.sort == 'desc' else 'ASC'}"
    if plan.limit:
        sql += f"\nLIMIT {plan.limit}"
    return sql


# ── AggregateBuilder ───────────────────────────────────────────────────────

def aggregate_builder(plan: Plan, axon: AxonLoader) -> str:
    """SELECT dims, COUNT/SUM(…) FROM … GROUP BY dims …"""
    base   = plan.measure_node.get("source", "")
    extra  = _collect_tables(plan, base)
    joins  = axon.build_joins(base, extra)

    dim_refs = [f"{dn['source']}.{dn['column']}" for _, dn in plan.dimensions]
    measure_expr = _measure_compute(plan.measure_key, plan.measure_node, axon)
    select_items = dim_refs + [measure_expr]

    sql = f"SELECT {', '.join(select_items)}\nFROM {base}"
    if joins:
        sql += "\n" + "\n".join(joins)

    where = build_where_clause(plan.filters, axon)
    if where:
        sql += f"\nWHERE {where}"

    if dim_refs:
        sql += f"\nGROUP BY {', '.join(dim_refs)}"

    if plan.sort:
        sql += f"\nORDER BY result {'DESC' if plan.sort == 'desc' else 'ASC'}"
    if plan.limit:
        sql += f"\nLIMIT {plan.limit}"
    return sql


# ── RatioBuilder ───────────────────────────────────────────────────────────

def ratio_builder(plan: Plan, axon: AxonLoader) -> str:
    """
    Builds:
        SELECT dims,
               ROUND(num_expr / NULLIF(den_expr, 0) * 100, 1) AS result
        FROM base
        GROUP BY dims
    """
    mn       = plan.measure_node
    num_key  = mn["numerator"]
    den_key  = mn["denominator"]
    mult     = mn.get("multiply", 1)
    rnd      = mn.get("round", 2)

    num_node = axon.measures.get(num_key, {})
    den_node = axon.measures.get(den_key, {})

    # Determine base table (prefer numerator's source)
    base = num_node.get("source") or den_node.get("source") or ""
    if not base:
        # fallback: pick from denominator
        base = den_node.get("source", "")

    # Collect all tables needed
    extra: set[str] = set()
    for _, dn in plan.dimensions:
        extra.add(dn["source"])
    for f in plan.filters:
        dn = f.get("dim_node", {})
        if dn.get("source"):
            extra.add(dn["source"])
    for node in (num_node, den_node):
        src = node.get("source", "")
        if src and src != base:
            extra.add(src)
    extra.discard(base)
    joins = axon.build_joins(base, extra)

    # Build CASE WHEN expressions for conditional counts
    def case_expr(m_key: str, m_node: dict) -> str:
        t   = m_node.get("type", "count")
        src = m_node.get("source", base)
        if t == "count":
            flt = m_node.get("filter")
            if flt:
                dim_key = flt["field"]
                dn      = axon.dimensions.get(dim_key, {})
                col_ref = f"{dn.get('source', src)}.{dn.get('column', dim_key)}"
                dtype   = dn.get("type", "string")
                val     = str(flt["val"]).replace("'", "''")
                if dtype in ("id", "number"):
                    cond = f"{col_ref} = {val}"
                else:
                    cond = f"LOWER({col_ref}) = LOWER('{val}')"
                return f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            return f"COUNT({src}.id)"
        return f"COUNT(*)"

    num_expr = case_expr(num_key, num_node)
    den_expr = case_expr(den_key, den_node)

    ratio_sql = f"ROUND({num_expr} / NULLIF({den_expr}, 0) * {mult}, {rnd})"
    dim_refs  = [f"{dn['source']}.{dn['column']}" for _, dn in plan.dimensions]
    items     = dim_refs + [f"{ratio_sql} AS result"]

    sql = f"SELECT {', '.join(items)}\nFROM {base}"
    if joins:
        sql += "\n" + "\n".join(joins)

    where = build_where_clause(plan.filters, axon)
    if where:
        sql += f"\nWHERE {where}"

    if dim_refs:
        sql += f"\nGROUP BY {', '.join(dim_refs)}"

    if plan.sort:
        sql += f"\nORDER BY result {'DESC' if plan.sort == 'desc' else 'ASC'}"
    if plan.limit:
        sql += f"\nLIMIT {plan.limit}"
    return sql


# ── WindowBuilder ──────────────────────────────────────────────────────────

def window_builder(plan: Plan, axon: AxonLoader) -> str:
    """
    Wraps an aggregate CTE with a RANK() OVER() window function.

    WITH base AS (
        SELECT dims, ratio_expr AS metric_val
        FROM ...
        GROUP BY dims
    )
    SELECT *, RANK() OVER (
        [PARTITION BY partition_dims]
        ORDER BY metric_val DESC
    ) AS result
    FROM base
    ORDER BY result ASC
    LIMIT n
    """
    mn       = plan.measure_node
    func     = mn.get("function", "RANK")
    order_m  = mn.get("order_by", {})
    ord_mk   = order_m.get("measure", "")
    ord_dir  = order_m.get("direction", "desc").upper()
    part_dim_keys = mn.get("partition_by", [])

    # Resolve the inner measure for the CTE
    inner_mk, inner_mn = axon.resolve_measure(ord_mk)
    if not inner_mk:
        raise ValueError(f"Window measure references unknown inner measure '{ord_mk}'")

    inner_type = inner_mn.get("type", "count")
    # For ratio inner measures, derive base from numerator's source
    if inner_type == "ratio":
        _num_src = axon.measures.get(inner_mn.get("numerator", ""), {}).get("source", "")
        _den_src = axon.measures.get(inner_mn.get("denominator", ""), {}).get("source", "")
        base = _num_src or _den_src or plan.measure_node.get("source", "")
    else:
        base = inner_mn.get("source") or plan.measure_node.get("source", "")

    # Extra tables for dims + filters + partition dims
    extra: set[str] = set()
    for _, dn in plan.dimensions:
        extra.add(dn["source"])
    for f in plan.filters:
        dn = f.get("dim_node", {})
        if dn.get("source"):
            extra.add(dn["source"])
    for pk in part_dim_keys:
        pdn = axon.dimensions.get(pk, {})
        if pdn.get("source"):
            extra.add(pdn.get("source"))
    extra.discard(base)
    joins = axon.build_joins(base, extra)

    # Build CTE select
    dim_refs  = [f"{dn['source']}.{dn['column']}" for _, dn in plan.dimensions]
    part_refs = [
        f"{axon.dimensions[pk]['source']}.{axon.dimensions[pk]['column']}"
        for pk in part_dim_keys if pk in axon.dimensions
    ]
    all_group_refs = list(dict.fromkeys(dim_refs + part_refs))  # deduped ordered

    if inner_type == "ratio":
        # Build ratio expr inline
        num_node = axon.measures.get(inner_mn["numerator"], {})
        den_node = axon.measures.get(inner_mn["denominator"], {})
        mult = inner_mn.get("multiply", 1)
        rnd  = inner_mn.get("round", 2)

        def _case(m_node: dict) -> str:
            t   = m_node.get("type", "count")
            src = m_node.get("source", base)
            flt = m_node.get("filter")
            if t == "count" and flt:
                dkey = flt["field"]
                dn   = axon.dimensions.get(dkey, {})
                cr   = f"{dn.get('source', src)}.{dn.get('column', dkey)}"
                dtype = dn.get("type", "string")
                val   = str(flt["val"]).replace("'", "''")
                cond  = f"{cr} = {val}" if dtype in ("id", "number") else f"LOWER({cr}) = LOWER('{val}')"
                return f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            return f"COUNT({src}.id)"

        metric_expr = f"ROUND({_case(num_node)} / NULLIF({_case(den_node)}, 0) * {mult}, {rnd})"
    else:
        src = inner_mn.get("source", base)
        flt = inner_mn.get("filter")
        if flt:
            dkey   = flt["field"]
            dn     = axon.dimensions.get(dkey, {})
            col_ref = f"{dn.get('source', src)}.{dn.get('column', dkey)}"
            dtype  = dn.get("type", "string")
            val    = str(flt["val"]).replace("'", "''")
            cond   = f"{col_ref} = {val}" if dtype in ("id", "number") else f"LOWER({col_ref}) = LOWER('{val}')"
            metric_expr = f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
        else:
            metric_expr = f"COUNT({src}.id)"

    cte_items = all_group_refs + [f"{metric_expr} AS metric_val"]
    cte_sql   = f"SELECT {', '.join(cte_items)}\nFROM {base}"
    if joins:
        cte_sql += "\n" + "\n".join(joins)

    where = build_where_clause(plan.filters, axon)
    if where:
        cte_sql += f"\nWHERE {where}"
    if all_group_refs:
        cte_sql += f"\nGROUP BY {', '.join(all_group_refs)}"

    # Outer window SELECT
    part_clause = ""
    if part_refs:
        part_clause = f"PARTITION BY {', '.join(part_refs)} "

    window_expr = f"{func}() OVER ({part_clause}ORDER BY metric_val {ord_dir}) AS result"

    outer_select = f"SELECT *, {window_expr}\nFROM base"
    if plan.sort:
        outer_select += f"\nORDER BY result {'ASC' if plan.sort == 'asc' else 'DESC'}"
    if plan.limit:
        outer_select += f"\nLIMIT {plan.limit}"

    return f"WITH base AS (\n{cte_sql}\n)\n{outer_select}"


# ── CompositeBuilder ───────────────────────────────────────────────────────

def composite_builder(plan: Plan, axon: AxonLoader) -> str:
    """
    Multi-CTE weighted score:

    WITH
      raw AS (SELECT dims, m1 AS v1, m2 AS v2 ... GROUP BY dims),
      normalised AS (SELECT *, v2/MAX(v2) OVER() AS v2_n ... FROM raw),
      scored AS (SELECT *, (w1*v1 + w2*v2_n) AS result FROM normalised)
    SELECT * FROM scored ORDER BY result DESC LIMIT n
    """
    formula   = plan.measure_node.get("formula", [])
    dim_nodes = plan.dimensions

    # Collect base table from first formula measure
    base = ""
    for item in formula:
        mn = axon.measures.get(item["measure"], {})
        if mn.get("source"):
            base = mn["source"]
            break

    extra: set[str] = set()
    for _, dn in dim_nodes:
        extra.add(dn["source"])
    for f in plan.filters:
        dn = f.get("dim_node", {})
        if dn.get("source"):
            extra.add(dn["source"])
    for item in formula:
        mn = axon.measures.get(item["measure"], {})
        src = mn.get("source", "")
        if src:
            extra.add(src)
    extra.discard(base)
    joins = axon.build_joins(base, extra)

    dim_refs = [f"{dn['source']}.{dn['column']}" for _, dn in dim_nodes]
    where    = build_where_clause(plan.filters, axon)

    # ── CTE 1: raw aggregates ──────────────────────────────────────────
    raw_cols: list[str] = list(dim_refs)
    norm_flags: dict[str, bool] = {}
    for i, item in enumerate(formula):
        mk  = item["measure"]
        mn  = axon.measures.get(mk, {})
        alias = f"v{i}"
        norm_flags[alias] = item.get("normalize", False)

        inner_type = mn.get("type", "count")
        if inner_type == "ratio":
            num_node = axon.measures.get(mn["numerator"], {})
            den_node = axon.measures.get(mn["denominator"], {})
            mult = mn.get("multiply", 1); rnd = mn.get("round", 2)

            def _case2(m: dict) -> str:
                src = m.get("source", base)
                flt = m.get("filter")
                if flt:
                    dkey = flt["field"]
                    dn   = axon.dimensions.get(dkey, {})
                    cr   = f"{dn.get('source', src)}.{dn.get('column', dkey)}"
                    dtype = dn.get("type", "string")
                    val   = str(flt["val"]).replace("'", "''")
                    cond  = f"{cr} = {val}" if dtype in ("id", "number") else f"LOWER({cr}) = LOWER('{val}')"
                    return f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
                return f"COUNT({src}.id)"

            expr = f"ROUND({_case2(num_node)} / NULLIF({_case2(den_node)}, 0) * {mult}, {rnd})"
        else:
            src = mn.get("source", base)
            flt = mn.get("filter")
            if flt:
                dkey   = flt["field"]
                dn_    = axon.dimensions.get(dkey, {})
                cr     = f"{dn_.get('source', src)}.{dn_.get('column', dkey)}"
                dtype  = dn_.get("type", "string")
                val    = str(flt["val"]).replace("'", "''")
                cond   = f"{cr} = {val}" if dtype in ("id", "number") else f"LOWER({cr}) = LOWER('{val}')"
                expr   = f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            else:
                expr = f"COUNT({src}.id)"

        raw_cols.append(f"{expr} AS {alias}")

    raw_cte = f"SELECT {', '.join(raw_cols)}\nFROM {base}"
    if joins:
        raw_cte += "\n" + "\n".join(joins)
    if where:
        raw_cte += f"\nWHERE {where}"
    if dim_refs:
        raw_cte += f"\nGROUP BY {', '.join(dim_refs)}"

    # ── CTE 2: normalise ──────────────────────────────────────────────
    norm_cols = ["*"]
    needs_norm = any(v for v in norm_flags.values())
    norm_cte = None
    if needs_norm:
        norm_cols = list(dim_refs)
        for i, item in enumerate(formula):
            alias = f"v{i}"
            if norm_flags[alias]:
                norm_cols.append(f"CASE WHEN MAX({alias}) OVER() = 0 THEN 0 ELSE {alias} / MAX({alias}) OVER() END AS {alias}")
            else:
                norm_cols.append(alias)
        norm_cte = f"SELECT {', '.join(norm_cols)}\nFROM raw"

    # ── CTE 3: score ──────────────────────────────────────────────────
    weight_parts = []
    for i, item in enumerate(formula):
        w     = item.get("weight", 1.0)
        alias = f"v{i}"
        weight_parts.append(f"{w} * {alias}")

    score_expr  = " + ".join(weight_parts)
    prev_cte    = "normalised" if needs_norm else "raw"
    score_items = (dim_refs if dim_refs else ["*"]) + [f"ROUND({score_expr}, 2) AS result"]
    score_cte   = f"SELECT {', '.join(score_items)}\nFROM {prev_cte}"

    # ── Assemble CTEs ─────────────────────────────────────────────────
    ctes = [f"raw AS (\n{raw_cte}\n)"]
    if norm_cte:
        ctes.append(f"normalised AS (\n{norm_cte}\n)")
    ctes.append(f"scored AS (\n{score_cte}\n)")

    outer = "SELECT *\nFROM scored"
    if plan.sort:
        outer += f"\nORDER BY result {'DESC' if plan.sort == 'desc' else 'ASC'}"
    else:
        outer += "\nORDER BY result DESC"
    if plan.limit:
        outer += f"\nLIMIT {plan.limit}"

    return "WITH\n" + ",\n".join(ctes) + "\n" + outer


# ── SubqueryBuilder ────────────────────────────────────────────────────────

def subquery_builder(plan: Plan, axon: AxonLoader) -> str:
    """Renders a scalar subquery measure (e.g. average tasks per employee)."""
    qspec = plan.measure_node.get("query", {})
    select_expr = qspec.get("select", "")
    from_spec   = qspec.get("from", {})

    if isinstance(from_spec, dict) and "subquery" in from_spec:
        sq      = from_spec["subquery"]
        sq_sel  = sq.get("select", "*")
        sq_from = sq.get("from", "")
        sq_grp  = sq.get("group_by", [])
        inner   = f"SELECT {sq_sel}\nFROM {sq_from}"
        if sq_grp:
            inner += f"\nGROUP BY {', '.join(sq_grp)}"
        sql = f"SELECT {select_expr}\nFROM (\n{inner}\n) AS _subq"
    else:
        from_table = from_spec if isinstance(from_spec, str) else ""
        sql = f"SELECT {select_expr}\nFROM {from_table}"

    return sql


# ── FilteredDimBuilder ─────────────────────────────────────────────────────

def filtered_dim_builder(plan: Plan, axon: AxonLoader) -> str:
    """
    Employees with above-average task counts via HAVING > (subquery).

    SELECT employee_name, COUNT(tasks.id) AS result
    FROM employees
    LEFT JOIN tasks ON ...
    GROUP BY employee_name
    HAVING COUNT(tasks.id) > (SELECT AVG(task_count) FROM (...))
    """
    mn         = plan.measure_node
    base_mk    = mn.get("base_measure", "total_tasks")
    having_spec = mn.get("having", {})
    group_keys  = mn.get("group_by", [])

    base_node  = axon.measures.get(base_mk, {})
    base       = base_node.get("source", "")
    base_expr  = f"COUNT({base}.id)"

    # Resolve group_by dims
    grp_dims = []
    for gk in group_keys:
        dn = axon.dimensions.get(gk, {})
        if dn:
            grp_dims.append((gk, dn))

    extra: set[str] = set()
    for _, dn in grp_dims:
        extra.add(dn["source"])
    extra.discard(base)
    joins = axon.build_joins(base, extra)

    grp_refs = [f"{dn['source']}.{dn['column']}" for _, dn in grp_dims]
    items    = grp_refs + [f"{base_expr} AS result"]

    sql = f"SELECT {', '.join(items)}\nFROM {base}"
    if joins:
        sql += "\n" + "\n".join(joins)

    where = build_where_clause(plan.filters, axon)
    if where:
        sql += f"\nWHERE {where}"
    if grp_refs:
        sql += f"\nGROUP BY {', '.join(grp_refs)}"

    # HAVING clause referencing threshold
    having_parts = []
    for op_key, ref_mk in having_spec.items():
        ref_node = axon.measures.get(ref_mk, {})
        ref_sql  = subquery_builder(
            Plan(plan_type="subquery", measure_key=ref_mk, measure_node=ref_node),
            axon
        )
        op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
        sql_op = op_map.get(op_key, ">")
        having_parts.append(f"{base_expr} {sql_op} ({ref_sql})")

    if having_parts:
        sql += f"\nHAVING {' AND '.join(having_parts)}"

    if plan.sort:
        sql += f"\nORDER BY result {'DESC' if plan.sort == 'desc' else 'ASC'}"
    if plan.limit:
        sql += f"\nLIMIT {plan.limit}"
    return sql


# ── RecipeBuilder ──────────────────────────────────────────────────────────

def recipe_builder(plan: Plan, axon: AxonLoader) -> str:
    """
    Chains recipe steps into CTEs.

    WITH
      base AS (SELECT ... GROUP BY ...),
      ranked AS (SELECT *, RANK() OVER (...) FROM base)
    SELECT * FROM ranked
    ORDER BY ...
    """
    recipe    = plan.recipe_node
    steps     = recipe.get("steps", [])
    output_id = recipe.get("output", steps[-1]["id"] if steps else "base")
    default_sort = recipe.get("default_sort", {})

    cte_parts: list[str] = []
    prev_cte_id: str     = ""

    for step in steps:
        sid          = step.get("id", "base")
        from_ref     = step.get("from", prev_cte_id) or ""
        source_table = step.get("source", "")
        step_measures = step.get("measures", [])
        step_select   = step.get("select", [])
        step_groupby  = step.get("group_by", [])
        self_join_spec = step.get("self_join")
        step_having   = step.get("having", {})

        if from_ref:
            # Chain from previous CTE
            select_items: list[str] = []
            for s in step_select:
                if s == "*":
                    select_items.append("*")
                else:
                    dk, dn = axon.resolve_dimension(s)
                    if dk and dn:
                        select_items.append(f"{dn['source']}.{dn['column']}")
                    else:
                        select_items.append(s)

            for mk in step_measures:
                m_key, m_node = axon.resolve_measure(mk)
                if not m_key:
                    continue
                m_type = m_node.get("type", "count")
                if m_type == "window":
                    func      = m_node.get("function", "RANK")
                    order_m   = m_node.get("order_by", {})
                    ord_dir   = order_m.get("direction", "desc").upper()
                    part_keys = m_node.get("partition_by", [])
                    part_cols = []
                    for pk in part_keys:
                        pdn = axon.dimensions.get(pk, {})
                        if pdn:
                            part_cols.append(f"{pdn['source']}.{pdn['column']}")
                    part_clause = f"PARTITION BY {', '.join(part_cols)} " if part_cols else ""
                    order_col   = order_m.get("measure", "metric_val")
                    # If ordering by a known measure alias resolve to 'result' / 'metric_val'
                    order_ref   = "metric_val" if order_col in axon.measures else order_col
                    select_items.append(
                        f"{func}() OVER ({part_clause}ORDER BY {order_ref} {ord_dir}) AS result"
                    )

            step_sql = f"SELECT {', '.join(select_items)}\nFROM {from_ref}"

        else:
            # First step: real table query
            actual_base = source_table or (
                axon.measures.get(step_measures[0], {}).get("source", "")
                if step_measures else ""
            )
            select_items = []
            for s in step_select:
                dk, dn = axon.resolve_dimension(s)
                if dk and dn:
                    select_items.append(f"{dn['source']}.{dn['column']}")
                else:
                    select_items.append(s)  # raw expression like "manager.emp_name AS manager_name"

            extra: set[str] = set()
            for s in step_select:
                dk, dn = axon.resolve_dimension(s)
                if dk and dn:
                    extra.add(dn["source"])
            for gb in step_groupby:
                dk, dn = axon.resolve_dimension(gb)
                if dk and dn:
                    extra.add(dn["source"])

            agg_exprs: list[str] = []
            for mk in step_measures:
                m_key, m_node = axon.resolve_measure(mk)
                if not m_key:
                    continue
                m_type = m_node.get("type", "count")
                if m_type == "count":
                    src = m_node.get("source", actual_base)
                    flt = m_node.get("filter")
                    if flt:
                        dkey = flt["field"]
                        dn_  = axon.dimensions.get(dkey, {})
                        cr   = f"{dn_.get('source', src)}.{dn_.get('column', dkey)}"
                        dtype= dn_.get("type","string")
                        val  = str(flt["val"]).replace("'", "''")
                        cond = f"{cr} = {val}" if dtype in ("id","number") else f"LOWER({cr}) = LOWER('{val}')"
                        agg_exprs.append(f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) AS {m_key}")
                    else:
                        extra.add(src)
                        agg_exprs.append(f"COUNT({src}.id) AS {m_key}")
                elif m_type == "ratio":
                    num_node = axon.measures.get(m_node["numerator"], {})
                    den_node = axon.measures.get(m_node["denominator"], {})
                    mult = m_node.get("multiply", 1); rnd = m_node.get("round", 2)

                    def _c(mn_: dict) -> str:
                        src_ = mn_.get("source", actual_base)
                        flt_ = mn_.get("filter")
                        if flt_:
                            dk__ = flt_["field"]
                            dn__ = axon.dimensions.get(dk__, {})
                            cr__ = f"{dn__.get('source', src_)}.{dn__.get('column', dk__)}"
                            dtype__ = dn__.get("type","string")
                            val__ = str(flt_["val"]).replace("'","''")
                            cond__ = f"{cr__} = {val__}" if dtype__ in ("id","number") else f"LOWER({cr__}) = LOWER('{val__}')"
                            return f"SUM(CASE WHEN {cond__} THEN 1 ELSE 0 END)"
                        extra.add(src_)
                        return f"COUNT({src_}.id)"

                    agg_exprs.append(
                        f"ROUND({_c(num_node)} / NULLIF({_c(den_node)}, 0) * {mult}, {rnd}) AS {m_key}"
                    )

            # Also add raw measures needed by later window steps
            all_items = select_items + agg_exprs

            extra.discard(actual_base)
            joins_list = axon.build_joins(actual_base, extra)

            # Self-join override
            if self_join_spec:
                alias = self_join_spec["alias"]
                on_   = self_join_spec.get("on") or self_join_spec.get(True, "")
                rel   = next(
                    (r for r in axon.relationships.values() if r.get("via_alias") == alias),
                    None,
                )
                jtype = rel["join"] if rel else "LEFT"
                joins_list = [f"{jtype} JOIN employees AS {alias} ON {on_}"]

            step_sql = f"SELECT {', '.join(all_items)}\nFROM {actual_base}"
            if joins_list:
                step_sql += "\n" + "\n".join(joins_list)

            grp_cols = []
            for gb in step_groupby:
                dk, dn = axon.resolve_dimension(gb)
                if dk and dn:
                    grp_cols.append(f"{dn['source']}.{dn['column']}")
                else:
                    grp_cols.append(gb)  # raw expression

            if grp_cols:
                step_sql += f"\nGROUP BY {', '.join(grp_cols)}"

            if step_having:
                having_parts = []
                for agg_alias, cond_spec in step_having.items():
                    # e.g. total_employees: {gte: 1}
                    for op_, val_ in cond_spec.items():
                        op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
                        sql_op = op_map.get(op_, ">")
                        # Find compute for this alias
                        agg_mn = axon.measures.get(agg_alias, {})
                        agg_src = agg_mn.get("source", actual_base)
                        having_parts.append(f"COUNT({agg_src}.id) {sql_op} {val_}")
                if having_parts:
                    step_sql += f"\nHAVING {' AND '.join(having_parts)}"

        cte_parts.append(f"{sid} AS (\n{step_sql}\n)")
        prev_cte_id = sid

    # Outer SELECT
    sort_field = plan.sort and "result" or default_sort.get("field", "result")
    sort_dir   = ("DESC" if plan.sort == "desc" else "ASC") if plan.sort else default_sort.get("direction", "asc").upper()

    outer = f"SELECT *\nFROM {output_id}\nORDER BY {sort_field} {sort_dir}"
    if plan.limit:
        outer += f"\nLIMIT {plan.limit}"

    return "WITH\n" + ",\n".join(cte_parts) + "\n" + outer


# ═══════════════════════════════════════════════════════════════════════════
# QUERY PLANNER  (dispatches to correct builder)
# ═══════════════════════════════════════════════════════════════════════════

class QueryPlanner:
    BUILDER_MAP = {
        "simple":       simple_builder,
        "aggregate":    aggregate_builder,
        "ratio":        ratio_builder,
        "window":       window_builder,
        "composite":    composite_builder,
        "subquery":     subquery_builder,
        "filtered_dim": filtered_dim_builder,
        "recipe":       recipe_builder,
    }

    def build(self, plan: Plan, axon: AxonLoader) -> str:
        builder = self.BUILDER_MAP.get(plan.plan_type)
        if not builder:
            raise ValueError(f"No builder for plan type '{plan.plan_type}'")
        return builder(plan, axon)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

class AxonCompiler:
    """
    One-stop compiler.

    Usage:
        compiler = AxonCompiler("axon.yaml")
        sql = compiler.compile(intent_dict)
    """

    def __init__(self, axon_path: str = "axon.yaml"):
        self.axon    = AxonLoader(axon_path)
        self.resolver = IntentResolver(self.axon)
        self.planner  = QueryPlanner()

    def compile(self, intent: dict) -> tuple[str, Plan]:
        """
        Compile an intent dict → (sql_string, plan).
        Raises ValueError for unresolvable measures/dimensions.
        """
        plan = self.resolver.resolve(intent)
        sql  = self.planner.build(plan, self.axon)
        return sql, plan

    def get_prompt_context(self) -> dict:
        """
        Returns dicts for use in the LLM decomposition prompt:
            - measure_keys:    list of all measure keys
            - dimension_keys:  list of all dimension keys
            - recipe_keys:     list of all recipe keys
            - synonym_guide:   string block for the prompt
        """
        lines = ["## MEASURE SYNONYMS\n"]
        for k, v in self.axon.measures.items():
            syns = v.get("synonyms", [])
            lines.append(f"  {k}: {', '.join(syns)}")

        lines.append("\n## DIMENSION SYNONYMS\n")
        for k, v in self.axon.dimensions.items():
            syns = v.get("synonyms", [])
            lines.append(f"  {k}: {', '.join(syns)}")

        lines.append("\n## RECIPE SYNONYMS\n")
        for k, v in self.axon.recipes.items():
            syns = v.get("synonyms", [])
            lines.append(f"  {k}: {', '.join(syns)}")

        return {
            "measure_keys":   list(self.axon.measures.keys()),
            "dimension_keys": list(self.axon.dimensions.keys()),
            "recipe_keys":    list(self.axon.recipes.keys()),
            "synonym_guide":  "\n".join(lines),
        }


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    compiler = AxonCompiler("axon.yaml")

    test_cases = [
        # simple list
        {"metric": "total_employees", "dimensions": ["employee_name"], "mode": "list"},
        # aggregate
        {"metric": "total_tasks", "dimensions": ["department_name"]},
        # ratio
        {"metric": "task_completion_rate", "dimensions": ["employee_name"], "sort": "desc", "limit": 5},
        # conditional count
        {"metric": "tasks_done", "dimensions": ["department_name"]},
        # window
        {"metric": "employee_rank_by_completion", "dimensions": ["employee_name"], "sort": "asc", "limit": 3},
        # composite
        {"metric": "employee_performance_score", "dimensions": ["employee_name"], "sort": "desc", "limit": 5},
        # subquery
        {"metric": "avg_tasks_per_employee"},
        # filtered_dim (above avg)
        {"metric": "employees_above_avg_tasks"},
        # recipe
        {"recipe": "employee_efficiency_report"},
        # recipe
        {"recipe": "department_health_report"},
        # recipe self-join
        {"recipe": "manager_span_of_control", "limit": 5},
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"INTENT: {json.dumps(tc)}")
        try:
            sql, plan = compiler.compile(tc)
            print(f"PLAN TYPE: {plan.plan_type}")
            print(f"SQL:\n{sql}")
        except Exception as e:
            print(f"ERROR: {e}")