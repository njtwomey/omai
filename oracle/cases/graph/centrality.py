"""Centralities against NetworkX: degree, closeness (hop and weighted), betweenness (normalised or not, weighted,
with endpoints), edge betweenness, and the power iterations (eigenvector, Katz, PageRank with personalisation, HITS),
run with tight tolerances so that the comparison tier is that of an iterative method (1e-8)."""

import networkx as nx
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.graphs import NxGraph, digraph, directed_family, edges_of, spec, undirected_family, weighted_family
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def vector(g: NxGraph, values: dict[int, float]) -> FloatArray:
    """A node-indexed dict as an array in node order."""
    return np.array([float(values[v]) for v in range(g.number_of_nodes())])


def edge_key(g: NxGraph, u: int, v: int) -> tuple[int, int]:
    """An edge as (source, target), ordered for undirected graphs."""
    return (u, v) if g.is_directed() else (min(u, v), max(u, v))


def connected(g: NxGraph) -> bool:
    """Connected, or strongly connected when directed."""
    return nx.is_strongly_connected(digraph(g)) if g.is_directed() else nx.is_connected(g)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family() + directed_family() + weighted_family():
        out.append(case(f"{label}/degree", "degree", [], vector(g, nx.degree_centrality(g)), **spec(g)))
        out.append(case(f"{label}/closeness", "closeness", [], vector(g, nx.closeness_centrality(g)), **spec(g)))
        out.append(
            case(
                f"{label}/closeness-weighted",
                "closenessWeighted",
                [],
                vector(g, nx.closeness_centrality(g, distance="weight")),
                **spec(g),
            )
        )
        for normalized in [True, False]:
            for endpoints in [False, True]:
                bc = nx.betweenness_centrality(g, normalized=normalized, endpoints=endpoints)
                out.append(
                    case(
                        f"{label}/betweenness-{normalized}-{endpoints}",
                        "betweenness",
                        [],
                        vector(g, bc),
                        rtol=1e-10,
                        **spec(g),
                        normalized=normalized,
                        endpoints=endpoints,
                    )
                )
        out.append(
            case(
                f"{label}/betweenness-weighted",
                "betweennessWeighted",
                [],
                vector(g, nx.betweenness_centrality(g, weight="weight")),
                rtol=1e-10,
                **spec(g),
            )
        )
        eb = nx.edge_betweenness_centrality(g)
        ebs = {edge_key(g, int(u), int(v)): float(x) for (u, v), x in eb.items()}
        out.append(
            case(
                f"{label}/edge-betweenness",
                "edgeBetweenness",
                [],
                np.array([ebs[(u, v)] for u, v, _ in edges_of(g)]),
                rtol=1e-10,
                **spec(g),
            )
        )
        weight = "weight"
        pr = nx.pagerank(g, tol=1e-13, max_iter=1000, weight=weight)
        out.append(case(f"{label}/pagerank", "pagerank", [], vector(g, pr), rtol=1e-8, atol=1e-10, **spec(g)))
        personal = {v: float(x) for v, x in enumerate(rng(len(label)).uniform(0.1, 1.0, size=g.number_of_nodes()))}
        prp = nx.pagerank(g, alpha=0.7, personalization=personal, tol=1e-13, max_iter=1000)
        out.append(
            case(
                f"{label}/pagerank-personal",
                "pagerankPersonal",
                [],
                vector(g, prp),
                rtol=1e-8,
                atol=1e-10,
                **spec(g),
                personalization=[personal[v] for v in range(g.number_of_nodes())],
            )
        )
        if g.number_of_edges() > 0:
            hubs, auths = nx.hits(g, tol=1e-14, max_iter=10000)
            out.append(
                case(
                    f"{label}/hits",
                    "hits",
                    [],
                    np.stack([vector(g, hubs), vector(g, auths)]),
                    rtol=1e-7,
                    atol=1e-9,
                    **spec(g),
                )
            )
        katz = nx.katz_centrality(g, alpha=0.05, tol=1e-13, max_iter=10000)
        out.append(case(f"{label}/katz", "katz", [], vector(g, katz), rtol=1e-8, atol=1e-10, **spec(g)))
        if connected(g) and not g.is_directed() and not nx.is_bipartite(g):
            ev = nx.eigenvector_centrality(g, tol=1e-13, max_iter=10000)
            out.append(case(f"{label}/eigenvector", "eigenvector", [], vector(g, ev), rtol=1e-7, atol=1e-9, **spec(g)))
    return out
