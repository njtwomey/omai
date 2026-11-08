"""Gamma-family functions against scipy.special."""

from functools import partial

import numpy as np
import scipy
import scipy.special as sp
from omai_oracle.fixture import Case
from omai_oracle.helpers import FloatArray, loguniform, reference, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def lfactorial(n: FloatArray) -> FloatArray:
    return sp.gammaln(n + 1)


def factorial(n: FloatArray) -> FloatArray:
    return sp.gamma(n + 1)


def lbinom(n: FloatArray, k: FloatArray) -> FloatArray:
    return sp.gammaln(n + 1) - sp.gammaln(k + 1) - sp.gammaln(n - k + 1)


def polygamma(order: int, x: FloatArray) -> FloatArray:
    return sp.polygamma(order, x)


def cases() -> list[Case]:
    g = rng(4)
    pos = np.concatenate([loguniform(g, 25, 0.01, 50), [0.5, 1.0, 2.0, 10.0]])
    neg = np.array([-0.5, -1.5, -2.3, -7.7])
    x = np.concatenate([pos, neg])
    a, b = loguniform(g, 25, 0.1, 100), loguniform(g, 25, 0.1, 100)
    n = np.array([0.0, 1, 5, 10, 20, 50, 2.5, 7.25])
    nn = np.array([5.0, 10, 30, 100, 7.5, 20.25, 52])
    kk = np.array([2.0, 3, 15, 50, 2.5, 0.5, 26])
    out = [
        reference("gamma", "gamma", sp.gamma, [x]),
        reference("lgamma", "lgamma", sp.gammaln, [np.concatenate([x, [1e4, 1e8]])], atol=1e-14),
        reference("rgamma", "rgamma", sp.rgamma, [np.concatenate([x, [-1.0, -2.0]])]),
        reference("digamma", "digamma", sp.digamma, [x], atol=1e-14),
        reference("beta", "beta", sp.beta, [a / 10, b / 10]),
        reference("lbeta", "lbeta", sp.betaln, [a, b], atol=1e-14),
        reference("factorial", "factorial", factorial, [n]),
        reference("lfactorial", "lfactorial", lfactorial, [n], atol=1e-14),
        reference("binom", "binom", sp.binom, [nn, kk]),
        reference("lbinom", "lbinom", lbinom, [nn, kk], atol=1e-12),
    ]
    for order in [1, 2, 3]:
        out.append(reference(f"polygamma/{order}", "polygamma", partial(polygamma, order), [pos], n=order))
    return out
