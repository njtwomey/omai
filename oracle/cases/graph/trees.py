"""Spanning trees against scipy.sparse.csgraph.minimum_spanning_tree and NetworkX (minimum and maximum spanning
trees and forests with distinct weights, so the trees are unique), and rooted-tree utilities against NetworkX (depths,
subtree sizes, DFS orders and lowest common ancestors)."""

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.graphs import NxGraph, distinct_weights, edges_of, relabelled, sparse, spec, undirected_family
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, (label, base) in enumerate(undirected_family()):
        g = distinct_weights(base, seed)
        mst = nx.minimum_spanning_tree(g, weight="weight")
        maxst = nx.maximum_spanning_tree(g, weight="weight")
        scipy_weight = float(minimum_spanning_tree(sparse(g)).sum())
        out.append(
            case(
                label,
                "spanning",
                [],
                np.array([scipy_weight, float(maxst.size(weight="weight"))]),
                **spec(g),
                treeEdges=[[u, v] for u, v, _ in edges_of(mst)],
                maxTreeEdges=[[u, v] for u, v, _ in edges_of(maxst)],
            )
        )
    for seed, (r, h) in enumerate([(2, 3), (3, 2)]):
        tree = relabelled(nx.balanced_tree(r, h))
        out.append(rooted(f"balanced-{r}-{h}", tree, 0, seed))
    for seed in range(3):
        tree = relabelled(nx.random_labeled_tree(15, seed=seed))
        out.append(rooted(f"random-tree-{seed}", tree, seed % 15, seed))
    return out


def rooted(label: str, tree: NxGraph, root: int, seed: int) -> Case:
    """Depths, subtree sizes, orders and LCAs of a tree rooted at `root`."""
    n = tree.number_of_nodes()
    directed = nx.bfs_tree(tree, root)
    depth = nx.single_source_shortest_path_length(tree, root)
    pairs = [[int(a), int(b)] for a, b in rng(seed).integers(0, n, size=(8, 2))]
    lca = [int(nx.lowest_common_ancestor(directed, a, b)) for a, b in pairs]
    sizes = [len(nx.descendants(directed, v)) + 1 for v in range(n)]
    return case(
        label,
        "rooted",
        [],
        np.array(0.0),
        **spec(tree),
        root=root,
        depth=[int(depth[v]) for v in range(n)],
        sizes=sizes,
        preorder=[int(v) for v in nx.dfs_preorder_nodes(tree, root)],
        postorder=[int(v) for v in nx.dfs_postorder_nodes(tree, root)],
        pairs=pairs,
        lca=lca,
    )
