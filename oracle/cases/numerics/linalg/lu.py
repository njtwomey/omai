"""LU factorisation, solve and inverse against numpy and scipy."""

import numpy as np
import scipy
import scipy.linalg as sl
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng, well_conditioned

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(22)
    a = g.normal(size=(5, 5))
    b = g.normal(size=(5, 2))
    batch = well_conditioned(g, 3, 3, (2,))
    bcols = g.normal(size=(2, 3, 1))
    perm, lower, upper = sl.lu(a)
    return [
        case("solve", "solve", [a, b], np.linalg.solve(a, b), rtol=1e-10),
        case("solve/vector", "solve", [a, b[:, 0]], np.linalg.solve(a, b[:, 0]), rtol=1e-10),
        case("solve/batch", "solve", [batch, bcols], np.linalg.solve(batch, bcols), rtol=1e-10),
        case("inv", "inv", [a], np.linalg.inv(a), rtol=1e-10),
        case("inv/batch", "inv", [batch], np.linalg.inv(batch), rtol=1e-10),
        case("lu/p", "lu", [a], perm, rtol=1e-12, part="p"),
        case("lu/l", "lu", [a], lower, rtol=1e-10, atol=1e-14, part="l"),
        case("lu/u", "lu", [a], upper, rtol=1e-10, atol=1e-14, part="u"),
    ]
