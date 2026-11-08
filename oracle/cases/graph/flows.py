"""Flows against NetworkX (maximum_flow_value, minimum_cut, min_cost_flow_cost, max_flow_min_cost,
hopcroft_karp_matching) and scipy.sparse.csgraph (maximum_flow, maximum_bipartite_matching). Flows and cuts are not
unique, so cases compare their values; the TypeScript tests also check feasibility."""

from collections.abc import Callable
from typing import cast

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sps
from networkx.algorithms import bipartite
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph, edges_of, gnp, spec, weighted_family
from omai_oracle.helpers import rng
from scipy.sparse.csgraph import maximum_bipartite_matching, maximum_flow

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def integer_csr(g: NxGraph) -> sps.csr_array:
    """Integer capacities as a CSR matrix (symmetric for undirected graphs), as scipy's maximum_flow needs."""
    a = np.asarray(nx.to_numpy_array(g, nodelist=sorted(g.nodes()), weight="weight"), dtype=np.int32)
    return sps.csr_array(a)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in weighted_family():
        n = g.number_of_nodes()
        s, t = 0, n - 1
        value = float(nx.maximum_flow_value(g, s, t, capacity="weight"))
        cut, _ = nx.minimum_cut(g, s, t, capacity="weight")
        scipy_value = float(maximum_flow(integer_csr(g), s, t).flow_value)
        out.append(case(label, "maxFlow", [], np.array([value, float(cut), scipy_value]), **spec(g), source=s, sink=t))
    for seed in range(4):
        g = gnp(10, 0.4, 20 + seed, directed=True, weights=(1, 9))
        edges = edges_of(g)
        costs = [float(c) for c in rng(seed).integers(-2 if seed % 2 == 1 else 1, 10, size=len(edges))]
        d: NxGraph = nx.DiGraph()
        d.add_nodes_from(range(10))
        for (u, v, w), c in zip(edges, costs, strict=True):
            d.add_edge(u, v, capacity=w, weight=c)
        flow = float(nx.maximum_flow_value(d, 0, 9))
        supply = min(flow, 6.0)
        demand = [0.0] * 10
        demand[0] = -supply
        demand[9] = supply
        for v, x in enumerate(demand):
            d.nodes[v]["demand"] = x
        cost = float(nx.min_cost_flow_cost(d))
        mfmc = cast(Callable[..., dict[int, dict[int, float]]], nx.max_flow_min_cost)(d, 0, 9)
        out.append(
            case(
                f"cost-{seed}",
                "minCostFlow",
                [],
                np.array([cost, float(nx.cost_of_flow(d, mfmc)), flow]),
                **spec(g),
                cost=costs,
                demand=demand,
            )
        )
    for seed, (n1, n2, p) in enumerate([(5, 6, 0.3), (8, 8, 0.2), (7, 4, 0.5), (10, 10, 0.15)]):
        b = bipartite.random_graph(n1, n2, p, seed=seed)
        matching = bipartite.hopcroft_karp_matching(b, top_nodes=range(n1))
        size = len(matching) // 2
        bi = sps.csr_array(
            np.asarray(bipartite.biadjacency_matrix(b, row_order=range(n1), column_order=range(n1, n1 + n2)).toarray())
        )
        scipy_size = int((maximum_bipartite_matching(bi, perm_type="column") >= 0).sum())
        edges = sorted([min(int(u), int(v)), max(int(u), int(v)), 1.0] for u, v in b.edges())
        out.append(
            case(
                f"matching-{seed}",
                "matching",
                [],
                np.array([float(size), float(scipy_size)]),
                n=n1 + n2,
                directed=False,
                edges=edges,
                left=list(range(n1)),
            )
        )
    return out
