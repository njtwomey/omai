"""Determinants, norms, condition numbers, rank and powers against numpy.linalg."""

from typing import Literal

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

# numpy.linalg.norm orders, with the labels the TypeScript test maps back to omai orders.
MATRIX_ORDERS: list[tuple[str, Literal["fro", "nuc"] | float]] = [
    ("fro", "fro"),
    ("nuc", "nuc"),
    ("1", 1),
    ("-1", -1),
    ("2", 2),
    ("-2", -2),
    ("inf", np.inf),
    ("-inf", -np.inf),
]
VECTOR_ORDERS: list[tuple[str, float]] = [("2", 2), ("1", 1), ("3", 3), ("0", 0), ("inf", np.inf), ("-inf", -np.inf)]


def cases() -> list[Case]:
    g = rng(23)
    a = g.normal(size=(4, 4))
    batch = g.normal(size=(3, 3, 3))
    rect = g.normal(size=(4, 3))
    v = g.normal(size=6)
    sign, logdet = np.linalg.slogdet(a)
    out = [
        case("det", "det", [a], np.linalg.det(a), rtol=1e-10),
        case("det/batch", "det", [batch], np.linalg.det(batch), rtol=1e-10),
        case("slogdet/sign", "slogdet", [a], sign, part="sign"),
        case("slogdet/logabsdet", "slogdet", [a], logdet, rtol=1e-10, part="logabsdet"),
        case("cond", "cond", [rect], np.linalg.cond(rect), rtol=1e-10),
        case("cond/1", "cond", [a], np.linalg.cond(a, 1), rtol=1e-10, p=1),
        case("matrixPower/3", "matrixPower", [a], np.linalg.matrix_power(a, 3), rtol=1e-10, k=3),
        case("matrixPower/-2", "matrixPower", [a], np.linalg.matrix_power(a, -2), rtol=1e-10, k=-2),
        case("matrixPower/0", "matrixPower", [a], np.linalg.matrix_power(a, 0), k=0),
    ]
    for label, o in MATRIX_ORDERS:
        out.append(case(f"norm/matrix/{label}", "norm", [rect], np.linalg.norm(rect, o), rtol=1e-10, ord=label))
    for label, o in VECTOR_ORDERS:
        out.append(case(f"norm/vector/{label}", "norm", [v], np.linalg.norm(v, o), rtol=1e-10, ord=label))
    low_rank = g.normal(size=(5, 2)) @ g.normal(size=(2, 4))
    out.append(case("matrixRank", "matrixRank", [low_rank], np.array(float(np.linalg.matrix_rank(low_rank)))))
    return out
