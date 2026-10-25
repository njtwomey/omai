"""Incomplete gamma and beta functions and their inverses against scipy.special."""

import numpy as np
import scipy
import scipy.special as sp
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import FloatArray, loguniform, reference, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


# Logs of the smaller tail directly, of the larger one through log1p of the other, so neither rounds to log(1).
def log_gammainc(a: FloatArray, x: FloatArray) -> FloatArray:
    p, q = sp.gammainc(a, x), sp.gammaincc(a, x)
    with np.errstate(divide="ignore"):  # the unselected branch may take log(0)
        return np.where(p < 0.5, np.log(p), np.log1p(-q))


def log_gammaincc(a: FloatArray, x: FloatArray) -> FloatArray:
    p, q = sp.gammainc(a, x), sp.gammaincc(a, x)
    with np.errstate(divide="ignore"):  # the unselected branch may take log(0)
        return np.where(q < 0.5, np.log(q), np.log1p(-p))


def cases() -> list[Case]:
    g = rng(5)
    a = loguniform(g, 30, 0.1, 100)
    x = a * loguniform(g, 30, 0.1, 5)
    p = np.concatenate([g.uniform(0.01, 0.99, 28), [1e-6, 0.999999]])
    ba, bb = loguniform(g, 30, 0.1, 100), loguniform(g, 30, 0.1, 100)
    bx = g.uniform(0.01, 0.99, 30)
    ints = np.array([1.0, 2, 3, 5, 1, 2])
    return [
        reference("gammainc", "gammainc", sp.gammainc, [a, x]),
        reference("gammaincc", "gammaincc", sp.gammaincc, [a, x]),
        reference("logGammainc", "logGammainc", log_gammainc, [a, x]),
        reference("logGammaincc", "logGammaincc", log_gammaincc, [a, x]),
        reference("gammaincinv", "gammaincinv", sp.gammaincinv, [a, p]),
        reference("gammainccinv", "gammainccinv", sp.gammainccinv, [a, p]),
        reference("betainc", "betainc", sp.betainc, [ba, bb, bx]),
        reference("betainc/integer-shapes", "betainc", sp.betainc, [ints, ints[::-1], bx[:6]]),
        reference("betaincinv", "betaincinv", sp.betaincinv, [ba, bb, p]),
    ]
