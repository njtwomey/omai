"""Traversals against scipy.sparse.csgraph (breadth_first_order, depth_first_order) and NetworkX (DFS orders and tree
edges, lexicographic topological sort, acyclicity, cycles, bipartiteness, k-hop neighbourhoods)."""

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.csgraph import breadth_first_order, depth_first_order
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.graphs import build, digraph, directed_family, ints, sparse, spec, undirected_family

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def predecessors(p: np.ndarray) -> list[int]:
    """scipy's predecessor array with its -9999 sentinel as -1."""
    return [int(x) if x >= 0 else -1 for x in p]


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family() + directed_family():
        a = sparse(g, weight=None)
        for source in [0, g.number_of_nodes() // 2]:
            order, pred = breadth_first_order(a, source, directed=True, return_predecessors=True)
            dorder, dpred = depth_first_order(a, source, directed=True, return_predecessors=True)
            hops = nx.single_source_shortest_path_length(g, source)
            tree = sorted(
                [int(u), int(v)] for u, v, kind in nx.dfs_labeled_edges(g, source) if kind == "forward" and u != v
            )
            out.append(
                case(
                    f"{label}/{source}",
                    "traversal",
                    [],
                    np.array(0.0),
                    **spec(g),
                    source=source,
                    bfsOrder=[int(x) for x in order],
                    bfsPredecessors=predecessors(pred),
                    distance=[int(hops.get(v, -1)) for v in range(g.number_of_nodes())],
                    dfsOrder=[int(x) for x in dorder],
                    dfsPredecessors=predecessors(dpred),
                    preorder=[int(v) for v in nx.dfs_preorder_nodes(g, source)],
                    postorder=[int(v) for v in nx.dfs_postorder_nodes(g, source)],
                    treeEdges=tree,
                    kHop2=sorted(int(v) for v in nx.single_source_shortest_path_length(g, source, cutoff=2)),
                )
            )
        has_cycle = True
        try:
            nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            has_cycle = False
        out.append(
            case(
                f"{label}/global",
                "global",
                [],
                np.array(0.0),
                **spec(g),
                hasCycle=has_cycle,
                bipartite=bool(nx.is_bipartite(g)),
                dag=bool(nx.is_directed_acyclic_graph(g)),
            )
        )
    for seed in range(3):
        dag = nx.gnp_random_graph(12, 0.3, seed=seed, directed=True)
        edges = [(int(u), int(v), 1.0) for u, v in dag.edges() if u < v]
        d = build(12, edges, True)
        out.append(
            case(
                f"dag-{seed}",
                "topologicalSort",
                [],
                np.array(0.0),
                **spec(d),
                order=ints(nx.lexicographical_topological_sort(digraph(d))),
            )
        )
    return out
