"""Distributional checks of the discrete samplers against scipy.stats: quantiles x_q and the CDF there.

As for the continuous samplers, the test checks the empirical CDF of many draws at each x_q against scipy's F(x_q).
"""

import math

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.distributions import Frozen
from whitebox_oracle.fixture import Case, encode, encode_params

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def quantile_case(name: str, fn: str, frozen: Frozen, **params: object) -> Case:
    """Rows x and F(x) at the quantile levels (repeated support points are kept; the test checks each)."""
    x = np.asarray(frozen.ppf(LEVELS), dtype=np.float64)
    f = np.asarray(frozen.cdf(x), dtype=np.float64)
    return Case(name=name, fn=fn, params=encode_params(params), output=encode(np.stack([x, f])))


def table_case(name: str, fn: str, weights: list[float]) -> Case:
    """A finite distribution by its direct definition: x_q = min{x : F(x) >= q}, with F the cumulative sum."""
    cdf = np.cumsum(np.array(weights)) / sum(weights)
    x = np.searchsorted(cdf, LEVELS).astype(np.float64)
    return Case(
        name=name, fn=fn, params=encode_params({"weights": weights}), output=encode(np.stack([x, cdf[x.astype(int)]]))
    )


def cases() -> list[Case]:
    weights = [1.0, 2.0, 7.0]
    out = [
        quantile_case("randint", "randint", stats.randint(-2, 5), low=-2, high=5),
        quantile_case("bernoulli", "bernoulli", stats.bernoulli(0.3), p=0.3),
        table_case("categorical", "categorical", weights),
        quantile_case("categorical/logits", "categoricalLogits", stats.bernoulli(0.75), logits=[0.0, math.log(3)]),
        table_case("aliasSample", "aliasSample", weights),
    ]
    for lam in [3.0, 29.5, 30.0, 80.0, 1000.0]:
        out.append(quantile_case(f"poisson/{lam}", "poisson", stats.poisson(lam), rate=lam))
    for n, p in [(20, 0.3), (20, 0.8), (59, 0.5), (61, 0.5), (1000, 0.3), (500, 0.97), (100000, 0.5), (10**7, 0.001)]:
        out.append(quantile_case(f"binomial/{n}/{p}", "binomial", stats.binom(n, p), n=n, p=p))
    for p in [0.2, 0.9]:
        out.append(quantile_case(f"geometric/{p}", "geometric", stats.geom(p), p=p))
    for n, ps, j in [(50, [0.2, 0.3, 0.5], 1), (50, [0.2, 0.3, 0.5], 2), (100000, [0.1, 0.6, 0.3], 1)]:
        out.append(
            quantile_case(
                f"multinomial/{n}/{j}", "multinomialMarginal", stats.binom(n, ps[j]), n=n, probs=ps, component=j
            )
        )
    return out
