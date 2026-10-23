"""Continuous distributions against scipy.stats."""

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.distributions import distribution_cases
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(30)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    x = g.normal(size=8)
    pos = np.exp(g.normal(size=8))
    unit = g.uniform(0.01, 0.99, 8)
    loc, scale = np.array([0.5, -1.0, 2.0]), np.array([1.0, 0.5, 3.0])
    out: list[Case] = []
    for i in range(3):
        a, s = loc[i : i + 1], scale[i : i + 1]
        out += distribution_cases(f"normal/{i}", "normal", [a, s], stats.norm(a, s), x, q)
        out += distribution_cases(f"logNormal/{i}", "logNormal", [a, s], stats.lognorm(s, scale=np.exp(a)), pos, q)
        out += distribution_cases(
            f"halfNormal/{i}", "halfNormal", [s], stats.halfnorm(scale=s), np.abs(x), q, skip=("logCdf", "logSf", "sf")
        )
    lo, hi = np.array([-1.0]), np.array([1.5])
    out += distribution_cases(
        "truncatedNormal",
        "truncatedNormal",
        [loc[:1], scale[:1], lo, hi],
        stats.truncnorm((lo - loc[:1]) / scale[:1], (hi - loc[:1]) / scale[:1], loc[:1], scale[:1]),
        np.clip(x, -1, 1.5),
        q,
        rtol=1e-9,
        skip=("sf", "logCdf", "logSf"),
    )
    for i, (alpha, rate) in enumerate([(0.5, 1.0), (2.0, 4.0), (30.0, 0.5)]):
        a, r = np.array([alpha]), np.array([rate])
        out += distribution_cases(f"gamma/{i}", "gamma", [a, r], stats.gamma(a, scale=1 / r), pos, q)
        out += distribution_cases(
            f"exponential/{i}", "exponential", [r], stats.expon(scale=1 / r), pos, q, skip=("logCdf",)
        )
    for i, (a_, b_) in enumerate([(0.5, 0.5), (2.0, 5.0), (30.0, 3.0)]):
        a, b = np.array([a_]), np.array([b_])
        out += distribution_cases(
            f"beta/{i}", "beta", [a, b], stats.beta(a, b), unit, q, skip=("sf", "logCdf", "logSf", "isf")
        )
    out += distribution_cases(
        "uniform",
        "uniform",
        [np.array([-1.0]), np.array([3.0])],
        stats.uniform(np.array([-1.0]), np.array([4.0])),
        g.uniform(-1, 3, 8),
        q,
        skip=("sf", "logCdf", "logSf", "isf"),
    )
    return out
