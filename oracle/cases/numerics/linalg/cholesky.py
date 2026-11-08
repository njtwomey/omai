"""Cholesky factorisation and solves against numpy and scipy."""

import numpy as np
import scipy
import scipy.linalg as sl
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, spd

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(21)
    a = spd(g, 5)
    batch = spd(g, 3, (2,))
    b = g.normal(size=5)
    factor = np.linalg.cholesky(a)
    return [
        case("cholesky", "cholesky", [a], factor, rtol=1e-10),
        case("cholesky/batch", "cholesky", [batch], np.linalg.cholesky(batch), rtol=1e-10),
        case("choSolve", "choSolve", [factor, b], sl.cho_solve((factor, True), b), rtol=1e-10),
        case("choleskyLogdet", "choleskyLogdet", [factor], np.linalg.slogdet(a)[1], rtol=1e-10),
    ]
