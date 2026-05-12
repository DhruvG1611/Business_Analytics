"""
graph_resolver.py
-----------------
Builds a logical query plan from resolved intent.
Key improvements:
  - compute_where split: never embeds WHERE inside compute expression
  - rank_per_group mode: emits partition_by for window CTE
  - Dimension inference fallback: adds default dims from metric join_path
    when LLM provides none
"""
import re
import networkx as nx
import config


class SchemaGraphResolver:
    def __init__(self):
        self.csm   = config.csm
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        for rel_name, rel in self.csm.get("relationships", {}).items():
            frm, to   = rel["from"], rel["to"]
            join_cond = rel["join"]
            weight    = 10 if frm == to else 1
            self.graph.add_edge(frm, to, join_on=join_cond, weight=weight)
            self.graph.add_edge(to, frm, join_on=join_cond, weight=weight)

    def find_shortest_path(self, start: str, target: str, forbidden: set = None) -> list:
        if start == target:
            return []
        G = self.graph.copy()
        if forbidden:
            G.remove_nodes_from([n for n in forbidden if n in G])
        try:
            path = nx.shortest_path(G, source=start, target=target, weight="weight")
            sequence = []
            for i in range(len(path) - 1):
                frm, to = path[i], path[i + 1]
                sequence.append((to, G[frm][to]["join_on"]))
            return sequence
        except nx.NetworkXNoPath:
            return None


def _extract_compute_columns(compute: str) -> set:
    refs = set(re.findall(r"\b(\w+\.\w+)\b", compute))
    return {ref.split(".")[1] for ref in refs} | refs


def _default_dims_for_metric(metric_node: dict) -> list[str]:
    """
    When the LLM provides no dimensions, infer safe defaults from the
    metric's join_path endpoint — the last table's id/name columns.
    Returns CSM dimension keys (not column refs).
    """
    join_path = metric_node.get("join_path", [])
    if not join_path:
        return []
    last_table = join_path[-1]
    # Find name-type dimensions on that table
    candidates = [
        k for k, v in config.csm.get("dimensions", {}).items()
        if v.get("source") == last_table and v.get("type") in ("string", "number")
           and "id" not in k.lower().split("_")[-1:]
    ]
    return candidates[:2]  # return at most 2 default dims


def rag_plus_plus_resolver(raw_intent: dict) -> dict:
    data    = raw_intent.get("intent", raw_intent)
    m_key   = data.get("metric", "")
    mode    = data.get("mode", "simple")
    metric_node = config.csm["metrics"].get(m_key)

    if not metric_node:
        raise ValueError(f"Metric '{m_key}' not found in CSM.")

    base_table    = metric_node["sources"][0]
    compute_expr  = metric_node.get("compute", "")
    compute_where = metric_node.get("compute_where", None)
    compute_cols  = _extract_compute_columns(compute_expr)

    # ── Dimensions ──────────────────────────────────────────────────────────
    requested_dims = data.get("dimensions") or []

    # Fallback: if LLM gave no dims and this isn't a bare count, infer defaults
    if not requested_dims and mode in ("ranked", "rank_per_group", "simple"):
        requested_dims = _default_dims_for_metric(metric_node)
        if requested_dims:
            print(f"  [GraphResolver] Inferred default dims: {requested_dims}")

    dim_nodes      = []
    required_tables = set()

    for d_id in requested_dims:
        node = config.csm["dimensions"].get(d_id)
        if not node:
            continue
        qualified = f"{node['source']}.{node['column']}"
        if qualified in compute_cols or node["column"] in compute_cols:
            continue
        dim_nodes.append((d_id, node))
        required_tables.add(node["source"])

    # ── Filters ─────────────────────────────────────────────────────────────
    valid_filters = []
    for f in data.get("filters", []):
        if not f.get("col_key"):
            continue
        f_dim = config.csm["dimensions"].get(f["col_key"])
        if f_dim:
            required_tables.add(f_dim["source"])
            valid_filters.append(f)

    # ── Graph join pathfinding ───────────────────────────────────────────────
    graph        = SchemaGraphResolver()
    active_joins = []
    joined       = {base_table}

    metric_join_path = metric_node.get("join_path", [])
    if metric_join_path:
        chain = (
            metric_join_path
            if metric_join_path[0] == base_table
            else [base_table] + metric_join_path
        )
        for i in range(len(chain) - 1):
            frm, to = chain[i], chain[i + 1]
            if to not in joined and graph.graph.has_edge(frm, to):
                active_joins.append(f"LEFT JOIN {to} ON {graph.graph[frm][to]['join_on']}")
                joined.add(to)
        forbidden = set(graph.graph.nodes) - set(chain) - required_tables
    else:
        forbidden = set()

    for table in required_tables:
        if table not in joined:
            path = graph.find_shortest_path(base_table, table, forbidden=forbidden)
            if path:
                for to_table, join_on in path:
                    if to_table not in joined:
                        active_joins.append(f"LEFT JOIN {to_table} ON {join_on}")
                        joined.add(to_table)

    # ── Assemble plan ────────────────────────────────────────────────────────
    dim_col_refs = [f"{node['source']}.{node['column']}" for _, node in dim_nodes]

    plan = {
        "select":           dim_col_refs + [f"{compute_expr} AS result"],
        "from":             base_table,
        "joins":            active_joins,
        "group_by":         dim_col_refs if dim_col_refs else [],
        "filters":          valid_filters,
        "compute_where":    compute_where,
        "having_threshold": data.get("having_threshold"),
        "sort":             data.get("sort"),
        "limit":            data.get("limit"),
        "mode":             mode,
    }

    # rank_per_group: add partition info for the window CTE in sql_compiler
    if mode == "rank_per_group" and dim_col_refs:
        # The grouping column is the "each X" dimension — heuristic: last dim added
        # from _infer_dimensions is the partition key, rest are the subject
        if mode == "rank_per_group" and dim_col_refs:
            plan["order_by_expr"] = compute_expr

            # Use partition_by from intent (set by RAG pattern) if available
            intent_partition = data.get("partition_by", [])
            if intent_partition:
                # Convert CSM dimension keys → qualified column refs
                resolved = []
                for dim_key in intent_partition:
                    node = config.csm["dimensions"].get(dim_key)
                    if node:
                        resolved.append(f"{node['source']}.{node['column']}")
                plan["partition_by"] = resolved if resolved else dim_col_refs[-1:]
            else:
                # Fallback heuristic — first dim is more likely the group key than last
                plan["partition_by"] = dim_col_refs[:1]

        

    return plan