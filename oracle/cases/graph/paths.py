"""Shortest paths against scipy.sparse.csgraph (dijkstra, bellman_ford, floyd_warshall, johnson, unweighted hop
counts) and NetworkX (A* lengths, Yen's k shortest simple paths by cost)."""

import itertools

import networkx as nx
import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import (
    Edge,
    NxGraph,
    build,
    dense,
    directed_family,
    gnp,
    sparse,
    spec,
    undirected_family,
    weighted_family,
)
from omai_oracle.helpers import rng
from scipy.sparse.csgraph import NegativeCycleError, bellman_ford, dijkstra, floyd_warshall, johnson

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def potential_graph(seed: int) -> NxGraph:
    """A directed graph with negative weights but no negative cycle: w(u, v) = c + h(u) - h(v) with c > 0."""
    base = gnp(10, 0.3, seed, directed=True, weights=(1, 6))
    h = rng(seed).integers(0, 8, size=10)
    edges: list[Edge] = [
        (u, v, w + float(h[u] - h[v])) for u, v, w in ((a, b, d["weight"]) for a, b, d in base.edges(data=True))
    ]
    return build(10, [e for e in edges if e[2] != 0], True)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in weighted_family() + undirected_family() + directed_family():
        a = sparse(g)
        directed = True  # undirected graphs are given as symmetric matrices
        out.append(case(f"{label}/dijkstra", "dijkstra", [], dijkstra(a, directed=directed), rtol=1e-12, **spec(g)))
        out.append(
            case(
                f"{label}/unweighted",
                "unweighted",
                [],
                dijkstra(a, directed=directed, unweighted=True),
                rtol=1e-12,
                **spec(g),
            )
        )
        out.append(case(f"{label}/floyd", "floydWarshall", [], floyd_warshall(a, directed=directed), **spec(g)))
        out.append(case(f"{label}/johnson", "johnson", [], johnson(a, directed=directed), **spec(g)))
        n = g.number_of_nodes()
        pairs = [(0, n - 1), (1, n // 2)]
        lengths: list[float] = []
        ksp: list[list[float]] = []
        for s, t in pairs:
            try:
                lengths.append(float(nx.astar_path_length(g, s, t, weight="weight")))
                paths = itertools.islice(nx.shortest_simple_paths(g, s, t, weight="weight"), 4)
                ksp.append([float(nx.path_weight(g, p, weight="weight")) for p in paths])
            except nx.NetworkXNoPath:
                lengths.append(float("inf"))
                ksp.append([])
        out.append(case(f"{label}/pairs", "pairs", [], np.array(lengths), **spec(g), pairs=pairs, kShortest=ksp))
    for seed in range(3):
        g = potential_graph(seed)
        out.append(
            case(f"negative-{seed}/bellman", "bellmanFord", [], bellman_ford(sparse(g), directed=True), **spec(g))
        )
        out.append(case(f"negative-{seed}/johnson", "johnson", [], johnson(sparse(g), directed=True), **spec(g)))
    cyc = build(4, [(0, 1, 1.0), (1, 2, -3.0), (2, 0, 1.0), (2, 3, 1.0)], True)
    negative = False
    try:
        bellman_ford(dense(cyc), directed=True)
    except NegativeCycleError:
        negative = True
    out.append(case("negative-cycle", "negativeCycle", [], np.array(0.0), **spec(cyc), negative=negative))
    return out
