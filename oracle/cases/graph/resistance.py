"""Effective resistance and path lengths against NetworkX: the resistance distance between every pair (edge weights as
conductances), the Kirchhoff index (effective_graph_resistance), the number of spanning trees (the matrix-tree
theorem, weighted and not) and the average shortest path length of connected graphs."""

from typing import cast

import networkx as nx
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.graphs import NxGraph, spec, undirected_family, weighted_family

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def connected_family() -> list[tuple[str, NxGraph]]:
    """The connected undirected graphs of the families: unweighted and weighted (conductances 1 to 9)."""
    graphs = undirected_family() + [(label, g) for label, g in weighted_family() if not g.is_directed()]
    return [(label, g) for label, g in graphs if nx.is_connected(g)]


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in connected_family():
        n = g.number_of_nodes()
        params = spec(g)
        dist = cast(dict[int, dict[int, float]], nx.resistance_distance(g, weight="weight", invert_weight=False))
        matrix = np.array([[dist[i][j] for j in range(n)] for i in range(n)], dtype=np.float64)
        out.append(case(f"{label}/resistance", "resistance", [], matrix, rtol=1e-9, atol=1e-10, **params))
        out.append(
            case(
                f"{label}/kirchhoff",
                "kirchhoff",
                [],
                np.array(nx.effective_graph_resistance(g, weight="weight", invert_weight=False)),
                rtol=1e-9,
                **params,
            )
        )
        out.append(
            case(
                f"{label}/trees",
                "trees",
                [],
                np.array([nx.number_of_spanning_trees(g), nx.number_of_spanning_trees(g, weight="weight")]),
                rtol=1e-8,
                **params,
            )
        )
        out.append(
            case(
                f"{label}/average-path-length",
                "averagePathLength",
                [],
                np.array(nx.average_shortest_path_length(g)),
                rtol=1e-12,
                **params,
            )
        )
    for n, k in [(20, 4), (30, 6), (25, 2)]:
        ring = nx.watts_strogatz_graph(n, k, 0)
        out.append(
            case(
                f"ring-{n}-{k}",
                "ring",
                [],
                np.array([nx.average_clustering(ring), nx.average_shortest_path_length(ring)]),
                n=n,
                k=k,
            )
        )
    return out
