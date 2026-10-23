"""Discrete distributions against scipy.stats (categorical against its direct numpy definition)."""

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.distributions import distribution_cases
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(31)
    out: list[Case] = []
    for i, p in enumerate([0.1, 0.5, 0.85]):
        out += distribution_cases(
            f"bernoulli/{i}",
            "bernoulli",
            [np.array([p])],
            stats.bernoulli(np.array([p])),
            np.array([0.0, 1.0]),
            np.array([0.05, 0.5, 0.95]),
            discrete=True,
            skip=("sf", "logCdf", "logSf", "ppf", "isf"),
        )
    counts = np.array([0.0, 1, 2, 3, 5, 8, 13])
    skip_discrete = ("sf", "logCdf", "logSf", "ppf", "isf")
    for i, (n, p) in enumerate([(10, 0.3), (25, 0.8)]):
        nn, pp = np.array([float(n)]), np.array([p])
        out += distribution_cases(
            f"binomial/{i}",
            "binomial",
            [nn, pp],
            stats.binom(nn, pp),
            counts,
            counts,
            discrete=True,
            skip=skip_discrete,
        )
    for i, lam in enumerate([0.7, 4.0, 30.0]):
        ll = np.array([lam])
        out += distribution_cases(
            f"poisson/{i}",
            "poisson",
            [ll],
            stats.poisson(ll),
            counts * (1 + i),
            counts,
            discrete=True,
            skip=skip_discrete,
        )
    for i, p in enumerate([0.2, 0.7]):
        pp = np.array([p])
        out += distribution_cases(
            f"geometric/{i}", "geometric", [pp], stats.geom(pp), counts + 1, counts, discrete=True, skip=skip_discrete
        )
    for i, (r, p) in enumerate([(3.0, 0.4), (0.7, 0.2)]):
        rr, pp = np.array([r]), np.array([p])
        out += distribution_cases(
            f"negativeBinomial/{i}",
            "negativeBinomial",
            [rr, pp],
            stats.nbinom(rr, pp),
            counts,
            counts,
            discrete=True,
            skip=skip_discrete,
        )
    probs = g.uniform(0.1, 1, 5)
    probs /= probs.sum()
    k = np.arange(5.0)
    x = np.array([0.0, 2.0, 4.0, 1.0])
    cdf = np.cumsum(probs)
    mean = float((k * probs).sum())
    out += [
        case("categorical/logProb", "categorical.logProb", [probs, x], np.log(probs[x.astype(int)]), rtol=1e-12),
        case("categorical/cdf", "categorical.cdf", [probs, x], cdf[x.astype(int)], rtol=1e-12),
        case("categorical/mean", "categorical.mean", [probs], np.array(mean), rtol=1e-12),
        case(
            "categorical/variance",
            "categorical.variance",
            [probs],
            np.array((k**2 * probs).sum() - mean**2),
            rtol=1e-12,
        ),
        case(
            "categorical/entropy", "categorical.entropy", [probs], np.array(-(probs * np.log(probs)).sum()), rtol=1e-12
        ),
    ]
    return out
