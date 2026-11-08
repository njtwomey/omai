"""Communities against NetworkX. Community labels and randomised searches are not unique, so cases compare canonical
forms: modularity values of given partitions (exactly), partitions as sorted sets where the answer is unambiguous
(Girvan-Newman levels, label propagation and Louvain on graphs of clear cliques, spectral bisection), and for Louvain
and Leiden on the karate club the modularity NetworkX reaches over several seeds."""

from collections.abc import Callable, Iterable
from typing import cast

import networkx as nx
import numpy as np
from networkx.algorithms import community as nxc
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import (
    NxGraph,
    build,
    directed_family,
    karate,
    relabelled,
    spec,
    undirected_family,
    weighted_family,
)
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def sorted_sets(sets: Iterable[Iterable[int]]) -> list[list[int]]:
    """Communities as sorted lists in lexicographic order."""
    return sorted(sorted(int(v) for v in s) for s in sets)


def leiden_modularity(g: NxGraph, seed: int) -> list[set[int]]:
    """NetworkX's Leiden with the modularity quality (its type stubs predate the `metric` argument)."""
    run = cast(Callable[..., list[set[int]]], nxc.leiden_communities)
    return run(g, seed=seed, metric="modularity")


def caveman() -> NxGraph:
    """Four 5-cliques joined in a ring by single edges: unambiguous communities."""
    return relabelled(nx.connected_caveman_graph(4, 5))


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family() + directed_family() + weighted_family():
        n = g.number_of_nodes()
        labels = [int(x) for x in rng(n).integers(0, 3, size=n)]
        parts = [{v for v in range(n) if labels[v] == c} for c in range(3)]
        parts = [p for p in parts if p]
        values = [float(nxc.modularity(g, parts, weight="weight", resolution=r)) for r in [1.0, 0.5, 2.0]]
        out.append(case(f"{label}/modularity", "modularity", [], np.array(values), **spec(g), labels=labels))
    for label, g in [("karate", karate()), ("barbell-5-0", relabelled(nx.barbell_graph(5, 0))), ("caveman", caveman())]:
        levels: list[list[list[int]]] = []
        for i, level in enumerate(nxc.girvan_newman(g)):
            levels.append(sorted_sets(list(level)))
            if i >= 2:
                break
        out.append(case(f"{label}/girvan-newman", "girvanNewman", [], np.array(0.0), **spec(g), levels=levels))
    for label, g in [("barbell-5-0", relabelled(nx.barbell_graph(5, 0))), ("caveman", caveman())]:
        lpa = sorted_sets(list(nxc.asyn_lpa_communities(g, seed=1)))
        louv = sorted_sets(nxc.louvain_communities(g, seed=1))
        leid = sorted_sets(leiden_modularity(g, 1))
        # The ring of caves has a repeated Fiedler value (no unique bisection), so only the barbell is bisected.
        bisect = sorted_sets(list(nx.spectral_bisection(g))) if label.startswith("barbell") else None
        out.append(
            case(
                f"{label}/clear",
                "clear",
                [],
                np.array(0.0),
                **spec(g),
                labelPropagation=lpa,
                louvain=louv,
                leiden=leid,
                bisection=bisect,
            )
        )
    g = karate()
    louvain = [float(nxc.modularity(g, nxc.louvain_communities(g, seed=s))) for s in range(10)]
    leiden = [float(nxc.modularity(g, leiden_modularity(g, s))) for s in range(10)]
    out.append(
        case(
            "karate/multilevel",
            "multilevel",
            [],
            np.array([min(louvain), max(louvain), min(leiden), max(leiden)]),
            **spec(g),
        )
    )
    bisect_graph = build(
        8, [(0, 1, 1), (1, 2, 1), (0, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (4, 6, 1), (6, 7, 1)], False
    )
    out.append(
        case(
            "lollipop/bisection",
            "bisection",
            [],
            np.array(0.0),
            **spec(bisect_graph),
            bisection=sorted_sets(list(nx.spectral_bisection(bisect_graph))),
            normalised=sorted_sets(list(nx.spectral_bisection(bisect_graph, normalized=True))),
        )
    )
    return out
