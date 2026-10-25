"""Binary-outcome distributions (dists/discrete/binary.ts) against scipy.stats."""

import numpy as np
import scipy
from mdml_oracle.distributions import distribution_cases
from mdml_oracle.fixture import Case
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    q = np.array([0.05, 0.3, 0.5, 0.77, 0.95])
    out: list[Case] = []
    for i, p in enumerate([0.1, 0.5, 0.85]):
        pp = np.array([p])
        out += distribution_cases(
            f"bernoulli/{i}",
            "bernoulli",
            [pp],
            stats.bernoulli(pp),
            np.array([0.0, 1.0, 2.0]),
            q,
            discrete=True,
            skip=("isf",),
        )
    counts = np.array([0.0, 1, 2, 3, 5, 8, 13])
    for i, (n, p) in enumerate([(10, 0.3), (25, 0.8)]):
        nn, pp = np.array([float(n)]), np.array([p])
        out += distribution_cases(
            f"binomial/{i}", "binomial", [nn, pp], stats.binom(nn, pp), counts, q, discrete=True, skip=("isf",)
        )
    for i, (n, a, b) in enumerate([(10, 2.0, 3.0), (25, 0.6, 0.9), (7, 5.0, 1.5)]):
        p = [np.array([float(n)]), np.array([a]), np.array([b])]
        out += distribution_cases(
            f"betaBinomial/{i}", "betaBinomial", p, stats.betabinom(*p), counts, q, discrete=True, skip=("isf",)
        )
    return out
