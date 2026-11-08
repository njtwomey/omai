"""Percolation bookkeeping against NetworkX: for a fixed order in which edges (bond percolation) or nodes (site
percolation) are switched on, the number of connected components and the size of the largest after each prefix of the
order, found by building the subgraph and calling connected_components."""

import networkx as nx
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph, edges_of, gnp, karate, spec
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def stats(g: "nx.Graph[int]") -> tuple[int, int]:
    """The number of components and the largest component's size (0 and 0 for the empty graph)."""
    sizes = [len(c) for c in nx.connected_components(g)]
    return len(sizes), max(sizes, default=0)


def graphs() -> list[tuple[str, NxGraph]]:
    return [
        ("gnp-14-0", gnp(14, 0.25, 0)),
        ("karate", karate()),
        ("grid-4x5", nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 5))),
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in graphs():
        n = g.number_of_nodes()
        edges = edges_of(g)
        edge_order = [int(i) for i in rng(n).permutation(len(edges))]
        bond: list[list[float]] = []
        for k in range(len(edges) + 1):
            h: NxGraph = nx.Graph()
            h.add_nodes_from(range(n))
            h.add_edges_from((edges[i][0], edges[i][1]) for i in edge_order[:k])
            count, largest = stats(h)
            bond.append([float(count), float(largest)])
        out.append(case(f"{label}/bond", "bond", [], np.array(bond), **spec(g), order=edge_order))
        node_order = [int(i) for i in rng(n + 100).permutation(n)]
        site: list[list[float]] = []
        for k in range(n + 1):
            site.append([float(x) for x in stats(g.subgraph(node_order[:k]))])
        out.append(case(f"{label}/site", "site", [], np.array(site), **spec(g), order=node_order))
    return out
