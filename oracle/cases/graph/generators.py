"""Deterministic graph families against NetworkX's generators, as sorted edge lists on relabelled nodes (grid nodes
(i, j) become i * n + j and hypercube nodes their binary codes, matching whitebox-js). Random models draw from
whitebox-js keys, so they are checked by statistical properties in the TypeScript tests instead."""

import networkx as nx
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.graphs import NxGraph, edges_of, karate, relabelled

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def grid(m: int, n: int, periodic: bool) -> NxGraph:
    g = nx.grid_2d_graph(m, n, periodic=periodic)
    mapping = {(i, j): i * n + j for i, j in g.nodes()}
    return relabelled(nx.relabel_nodes(g, mapping).subgraph(range(m * n)).copy())


def hypercube(d: int) -> NxGraph:
    g = nx.hypercube_graph(d)
    mapping = {node: sum(int(b) << k for k, b in enumerate(node)) for node in g.nodes()}
    h = nx.relabel_nodes(g, mapping)
    return relabelled(h.subgraph(range(2**d)).copy())


def families() -> "list[tuple[str, str, list[int], NxGraph]]":
    """(case name, generator, arguments, NetworkX graph)."""
    return [
        ("complete-5", "completeGraph", [5], nx.complete_graph(5)),
        ("complete-directed-4", "completeDigraph", [4], nx.complete_graph(4, create_using=nx.DiGraph)),
        ("path-6", "pathGraph", [6], nx.path_graph(6)),
        ("cycle-7", "cycleGraph", [7], nx.cycle_graph(7)),
        ("star-4", "starGraph", [4], nx.star_graph(4)),
        ("wheel-6", "wheelGraph", [6], nx.wheel_graph(6)),
        ("grid-3x4", "gridGraph", [3, 4], grid(3, 4, False)),
        ("torus-3x4", "torus", [3, 4], grid(3, 4, True)),
        ("hypercube-3", "hypercubeGraph", [3], hypercube(3)),
        ("bipartite-3-4", "completeBipartiteGraph", [3, 4], nx.complete_bipartite_graph(3, 4)),
        ("balanced-2-3", "balancedTree", [2, 3], nx.balanced_tree(2, 3)),
        ("balanced-3-2", "balancedTree", [3, 2], nx.balanced_tree(3, 2)),
        ("barbell-4-2", "barbellGraph", [4, 2], nx.barbell_graph(4, 2)),
        ("petersen", "petersenGraph", [], nx.petersen_graph()),
        ("karate", "karateClubGraph", [], karate()),
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, fn, args, g in families():
        h = relabelled(g)
        out.append(
            case(
                name,
                fn,
                [],
                np.array(0.0),
                args=args,
                n=h.number_of_nodes(),
                directed=h.is_directed(),
                edges=[[u, v] for u, v, _ in edges_of(h)],
            )
        )
    return out
