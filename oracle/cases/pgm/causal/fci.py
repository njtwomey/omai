"""FCI against causal-learn (fci with the Fisher z test): the PAG's edge marks on seeded linear-Gaussian data from
graphs with a latent confounder (a bidirected edge) and with colliders whose orientations propagate."""

from importlib.metadata import version
from typing import Any, cast

import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"causal-learn": version("causal-learn")}

MARKS: dict[int, str] = {-1: "tail", 1: "arrow", 2: "circle"}


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    g = rng(83)
    # Each: observed structure (name, [(parent, weight)]) where names starting with "L" are latent (dropped).
    graphs: list[list[tuple[str, list[tuple[str, float]]]]] = [
        [("a", []), ("d", []), ("L", []), ("b", [("a", 0.9), ("L", 0.8)]), ("c", [("d", 0.9), ("L", 0.8)])],
        [("x1", []), ("x2", []), ("x3", [("x1", 0.8), ("x2", 0.8)]), ("x4", [("x3", 0.9)]), ("x5", [("x4", 0.8)])],
        [
            ("a", []),
            ("b", []),
            ("L", []),
            ("c", [("a", 0.8), ("b", 0.8)]),
            ("d", [("c", 0.8), ("L", 0.9)]),
            ("e", [("L", 0.9)]),
        ],
    ]
    for t, structure in enumerate(graphs):
        n = 2000
        cols: dict[str, FloatArray] = {}
        for v, parents in structure:
            cols[v] = g.normal(size=n) + sum((w * cols[p] for p, w in parents), np.zeros(n))
        names = [v for v, _ in structure if not v.startswith("L")]
        x = np.column_stack([cols[v] for v in names])
        graph, _ = fci(x, fisherz, 0.05, show_progress=False, node_names=names)
        m = np.asarray(cast(Any, graph).graph)  # a GeneralGraph
        edges = [
            {"a": names[i], "b": names[j], "markA": MARKS[int(m[i, j])], "markB": MARKS[int(m[j, i])]}
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if m[i, j] != 0
        ]
        rows = [{v: float(c) for v, c in zip(names, r, strict=True)} for r in x]
        out.append(case(f"fci/{t}", "fci", [], zero, data=rows, edges=edges))
    return out
