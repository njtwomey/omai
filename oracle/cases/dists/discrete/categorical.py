"""Categorical distributions (dists/discrete/categorical.ts) against scipy.stats and their numpy definitions."""

import numpy as np
import scipy
from scipy import special, stats
from whitebox_oracle.distributions import distribution_cases, method_case
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, Unary, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def ordinal_probs(eta: float, cuts: FloatArray, cdf: Unary) -> FloatArray:
    """P(Y = k) for a cumulative-link model with P(Y <= k) = cdf(cut_k - eta)."""
    upper = np.concatenate([cdf(cuts - eta), [1.0]])
    lower = np.concatenate([[0.0], cdf(cuts - eta)])
    return np.asarray(upper - lower, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(31)
    probs = g.uniform(0.1, 1, 5)
    probs /= probs.sum()
    k = np.arange(5.0)
    x = np.array([0.0, 2.0, 4.0, 1.0])
    cdf = np.cumsum(probs)
    mean = float((k * probs).sum())
    out: list[Case] = [
        case("categorical/logProb", "categorical.logProb", [probs, x], np.log(probs[x.astype(int)]), arity=1),
        case("categorical/cdf", "categorical.cdf", [probs, x], cdf[x.astype(int)], arity=1),
        case("categorical/sf", "categorical.sf", [probs, x], 1 - cdf[x.astype(int)], arity=1),
        case(
            "categorical/ppf",
            "categorical.ppf",
            [probs, np.array([0.05, 0.5, 0.99])],
            np.searchsorted(cdf, np.array([0.05, 0.5, 0.99])).astype(np.float64),
            arity=1,
        ),
        case("categorical/mean", "categorical.mean", [probs], np.array(mean), arity=1),
        case(
            "categorical/variance", "categorical.variance", [probs], np.array((k**2 * probs).sum() - mean**2), arity=1
        ),
        case("categorical/entropy", "categorical.entropy", [probs], np.array(-(probs * np.log(probs)).sum()), arity=1),
    ]
    onehot = np.eye(5)[[1, 3, 4]]
    out += [
        case(
            "oneHotCategorical/logProb", "oneHotCategorical.logProb", [probs, onehot], np.log(probs[[1, 3, 4]]), arity=1
        ),
        case("oneHotCategorical/variance", "oneHotCategorical.variance", [probs], probs * (1 - probs), arity=1),
    ]
    q = np.array([0.05, 0.3, 0.5, 0.77, 0.95])
    for i, (lo, hi) in enumerate([(1.0, 6.0), (-3.0, 4.0)]):
        p = [np.array([lo]), np.array([hi])]
        out += distribution_cases(
            f"discreteUniform/{i}",
            "discreteUniform",
            p,
            stats.randint(p[0], p[1] + 1),
            np.arange(lo - 1, hi + 2),
            q,
            discrete=True,
            skip=("isf",),
        )
    cuts = np.array([-1.0, 0.3, 1.5])
    for i, eta in enumerate([0.0, 0.8, -1.2]):
        params = [np.array(eta), cuts]
        levels = np.arange(4.0)
        for name, fn in [("orderedLogistic", special.expit), ("orderedProbit", special.ndtr)]:
            pr = ordinal_probs(eta, cuts, fn)
            out.append(method_case(f"{name}/{i}", name, "logProb", params, levels, np.log(pr), rtol=1e-12))
            out.append(method_case(f"{name}/{i}", name, "cdf", params, levels, np.cumsum(pr), rtol=1e-12))
    return out
