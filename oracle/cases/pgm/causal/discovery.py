"""Causal discovery against causal-learn: GES with the linear-Gaussian BIC (ges, local_score_BIC) and DirectLiNGAM's
causal order, on seeded linear models (Gaussian noise for GES, uniform noise for LiNGAM)."""

from importlib.metadata import version
from typing import cast

import numpy as np
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.search.ScoreBased.GES import ges
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"causal-learn": version("causal-learn")}


def rows_of(x: FloatArray, names: list[str]) -> list[dict[str, float]]:
    return [{n: float(v) for n, v in zip(names, r, strict=True)} for r in x]


def cpdag_edges(graph: np.ndarray, names: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """causal-learn's endpoint matrix (graph[i, j] = -1, graph[j, i] = 1 for i -> j; both -1 for i - j) as edges."""
    directed: list[list[str]] = []
    undirected: list[list[str]] = []
    d = len(names)
    for i in range(d):
        for j in range(d):
            if graph[i, j] == -1 and graph[j, i] == 1:
                directed.append([names[i], names[j]])
            elif i < j and graph[i, j] == -1 and graph[j, i] == -1:
                undirected.append([names[i], names[j]])
    return sorted(directed), sorted(undirected)


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    g = rng(71)
    for t, (n, structure) in enumerate(
        [
            (600, [("a", []), ("b", []), ("c", [("a", 0.8), ("b", -0.7)]), ("d", [("c", 0.9)]), ("e", [("d", -0.6)])]),
            (800, [("a", []), ("b", [("a", 1.0)]), ("c", [("a", 0.6)]), ("d", [("b", 0.8), ("c", 0.7)])]),
        ]
    ):
        names = [v for v, _ in structure]
        cols: dict[str, FloatArray] = {}
        for v, parents in structure:
            cols[v] = g.normal(size=n) + sum((w * cols[p] for p, w in parents), np.zeros(n))
        x = np.column_stack([cols[v] for v in names])
        record = ges(x, score_func="local_score_BIC")
        directed, undirected = cpdag_edges(np.asarray(record["G"].graph), names)
        out.append(case(f"ges/{t}", "ges", [], zero, data=rows_of(x, names), directed=directed, undirected=undirected))
    for t, n in enumerate([500, 1000]):
        names = ["x0", "x1", "x2", "x3"]
        e = g.uniform(-1, 1, size=(n, 4))
        x0 = e[:, 0]
        x2 = 0.9 * x0 + e[:, 2]
        x1 = -0.8 * x2 + 0.5 * x0 + e[:, 1]
        x3 = 0.7 * x1 + e[:, 3]
        x = np.column_stack([x0, x1, x2, x3])
        model = DirectLiNGAM()
        model.fit(x)
        order = [names[int(i)] for i in cast(list[int], model.causal_order_)]
        out.append(case(f"lingam/{t}", "lingam", [], zero, data=rows_of(x, names), order=order))
    return out
