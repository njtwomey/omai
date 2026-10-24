"""Components against NetworkX and scipy.sparse.csgraph: connected, weakly and strongly connected components,
condensations, articulation points, bridges and biconnected components, all as sorted node sets."""

from collections.abc import Iterable

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.graphs import NxGraph, directed_family, ints, sparse, spec, undirected_family

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def sorted_sets(sets: Iterable[Iterable[int]]) -> list[list[int]]:
    """Node sets as sorted lists, in lexicographic order."""
    return sorted(sorted(int(v) for v in s) for s in sets)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family():
        count, _ = connected_components(sparse(g, weight=None), directed=False)
        blocks = [ints(b) for b in nx.biconnected_components(g)]
        block_edges = sorted(
            sorted(sorted([int(u), int(v)]) for u, v in edges) for edges in nx.biconnected_component_edges(g)
        )
        out.append(
            case(
                label,
                "undirected",
                [],
                np.array(0.0),
                **spec(g),
                count=int(count),
                components=sorted_sets(list(nx.connected_components(g))),
                articulation=sorted(int(v) for v in nx.articulation_points(g)),
                bridges=sorted(sorted([int(u), int(v)]) for u, v in nx.bridges(g)),
                blocks=sorted_sets(blocks),
                blockEdges=block_edges,
            )
        )
    for label, g in directed_family():
        d: NxGraph = nx.DiGraph(g)
        scc = sorted_sets(list(nx.strongly_connected_components(d)))
        c = nx.condensation(d)
        members = {int(k): sorted(int(v) for v in c.nodes[k]["members"]) for k in c.nodes()}
        dag_edges = sorted([members[int(a)], members[int(b)]] for a, b in c.edges())
        strong, _ = connected_components(sparse(g, weight=None), directed=True, connection="strong")
        out.append(
            case(
                label,
                "directed",
                [],
                np.array(0.0),
                **spec(g),
                strongCount=int(strong),
                strong=scc,
                weak=sorted_sets(list(nx.weakly_connected_components(d))),
                condensationEdges=dag_edges,
            )
        )
    return out
