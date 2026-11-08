"""Shared pieces for the graph case modules: seeded example graphs on nodes 0 .. n - 1 built in sorted edge order (so
NetworkX's adjacency order matches omai's sorted adjacency), their encoding for fixtures, and dense and sparse
matrices for scipy.sparse.csgraph."""

from collections.abc import Iterable
from typing import Any, TypedDict, cast

import networkx as nx
import numpy as np
import scipy.sparse as sps

from .helpers import FloatArray, rng

type Edge = tuple[int, int, float]
"""An edge as fixtures store it: source, target, weight."""

type NxGraph = nx.Graph[Any]
"""A NetworkX graph or digraph on integer nodes (the stubs' node type is invariant, so it is left open)."""


class GraphSpec(TypedDict):
    """A graph as fixture parameters."""

    n: int
    directed: bool
    edges: list[list[float]]


def build(n: int, edges: list[Edge], directed: bool) -> NxGraph:
    """A NetworkX graph on nodes 0 .. n - 1 with the edges added in sorted order."""
    g: NxGraph = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(n))
    canon = sorted(((u, v, w) if directed or u <= v else (v, u, w)) for u, v, w in edges)
    for u, v, w in canon:
        g.add_edge(u, v, weight=w)
    return g


def edges_of(g: NxGraph) -> list[Edge]:
    """The edges of a graph on integer nodes, sorted, with their weights (1 when absent)."""
    out: list[Edge] = []
    for u, v, d in g.edges(data=True):
        a, b = (u, v) if g.is_directed() or u <= v else (v, u)
        out.append((int(a), int(b), float(d.get("weight", 1.0))))
    return sorted(out)


def relabelled(g: NxGraph) -> NxGraph:
    """A copy on nodes 0 .. n - 1 (in the graph's node order) rebuilt in sorted edge order."""
    h = nx.convert_node_labels_to_integers(g)
    return build(h.number_of_nodes(), edges_of(h), h.is_directed())


def spec(g: NxGraph) -> GraphSpec:
    """The graph as fixture parameters: `n`, `directed` and `edges` as [source, target, weight] triples."""
    return {
        "n": g.number_of_nodes(),
        "directed": g.is_directed(),
        "edges": [[float(u), float(v), w] for u, v, w in edges_of(g)],
    }


def dense(g: NxGraph, weight: str | None = "weight") -> FloatArray:
    """The dense adjacency matrix in node order."""
    return np.asarray(nx.to_numpy_array(g, nodelist=sorted(g.nodes()), weight=cast(str, weight)), dtype=np.float64)


def sparse(g: NxGraph, weight: str | None = "weight") -> sps.csr_array:
    """The sparse adjacency matrix in node order (symmetric for undirected graphs)."""
    return sps.csr_array(dense(g, weight))


def gnp(n: int, p: float, seed: int, directed: bool = False, weights: tuple[int, int] | None = None) -> NxGraph:
    """A seeded G(n, p) graph, optionally with integer weights drawn uniformly from [lo, hi]."""
    base = nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    g = rng(seed)
    edges: list[Edge] = [
        (int(u), int(v), 1.0 if weights is None else float(g.integers(weights[0], weights[1] + 1)))
        for u, v in base.edges()
    ]
    return build(n, edges, directed)


def distinct_weights(g: NxGraph, seed: int) -> NxGraph:
    """The same graph with distinct weights 1 .. m in a random order (so minimum spanning trees are unique)."""
    edges = edges_of(g)
    order = rng(seed).permutation(len(edges))
    return build(
        g.number_of_nodes(), [(u, v, float(order[i] + 1)) for i, (u, v, _) in enumerate(edges)], g.is_directed()
    )


def karate() -> NxGraph:
    """Zachary's karate club, unweighted."""
    return build(34, [(int(u), int(v), 1.0) for u, v in nx.karate_club_graph().edges()], False)


def undirected_family() -> list[tuple[str, NxGraph]]:
    """Small undirected test graphs: random, structured and disconnected."""
    out: list[tuple[str, NxGraph]] = [(f"gnp-12-{seed}", gnp(12, 0.3, seed)) for seed in range(3)]
    out.append(("karate", karate()))
    out.append(("petersen", relabelled(nx.petersen_graph())))
    out.append(("grid-3x4", relabelled(nx.grid_2d_graph(3, 4))))
    out.append(("barbell-4-2", relabelled(nx.barbell_graph(4, 2))))
    out.append(("tree-2-3", relabelled(nx.balanced_tree(2, 3))))
    out.append(("disconnected", relabelled(nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(3)))))
    return out


def directed_family() -> list[tuple[str, NxGraph]]:
    """Small directed test graphs."""
    out: list[tuple[str, NxGraph]] = [(f"digraph-10-{seed}", gnp(10, 0.25, seed, directed=True)) for seed in range(3)]
    out.append(
        ("cycle-chain", build(6, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 3, 1)], True))
    )
    return out


def weighted_family() -> list[tuple[str, NxGraph]]:
    """Weighted test graphs, undirected and directed, with integer weights in [1, 9]."""
    out: list[tuple[str, NxGraph]] = [(f"weighted-12-{seed}", gnp(12, 0.35, seed, weights=(1, 9))) for seed in range(3)]
    out += [(f"weighted-digraph-10-{seed}", gnp(10, 0.3, seed, directed=True, weights=(1, 9))) for seed in range(2)]
    return out


def ints(xs: Iterable[object]) -> list[int]:
    """Node labels (typed as hashables by the stubs) as ints."""
    return [int(cast(int, x)) for x in xs]


def digraph(g: NxGraph) -> "nx.DiGraph[Any]":
    """A graph known to be directed, typed as a DiGraph."""
    return cast("nx.DiGraph[Any]", g)
