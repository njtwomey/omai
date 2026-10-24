"""Distributional checks of the continuous samplers against scipy.stats: quantiles x_q and the CDF there.

The samplers cannot reproduce numpy's streams, so each case gives points x (scipy's `ppf` at fixed levels) and F(x);
the test draws many samples and checks the empirical CDF at x against F(x) within a few standard errors.
"""

import math

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.distributions import Frozen
from whitebox_oracle.fixture import Case, encode, encode_params

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def quantile_case(name: str, fn: str, frozen: Frozen, **params: object) -> Case:
    """Rows x and F(x) at the quantile levels."""
    x = np.asarray(frozen.ppf(LEVELS), dtype=np.float64)
    f = np.asarray(frozen.cdf(x), dtype=np.float64)
    return Case(name=name, fn=fn, params=encode_params(params), output=encode(np.stack([x, f])))


def cases() -> list[Case]:
    out = [
        quantile_case("uniform", "uniform", stats.uniform(-1, 4), low=-1.0, high=3.0),
        quantile_case("normal", "normal", stats.norm(1, 2), loc=1.0, scale=2.0),
        quantile_case("exponential", "exponential", stats.expon(scale=0.5), rate=2.0),
        quantile_case("laplace", "laplace", stats.laplace(1, 2), loc=1.0, scale=2.0),
        quantile_case("logistic", "logistic", stats.logistic(0, 1.5), loc=0.0, scale=1.5),
        quantile_case("gumbel", "gumbel", stats.gumbel_r(0.5, 2), loc=0.5, scale=2.0),
        quantile_case("cauchy", "cauchy", stats.cauchy(0, 1), loc=0.0, scale=1.0),
        quantile_case("chisquare", "chisquare", stats.chi2(3), df=3.0),
        quantile_case("studentT", "studentT", stats.t(4), df=4.0),
    ]
    inf = math.inf
    for lo, hi in [
        (-1, 2),
        (2, inf),
        (-inf, -3),
        (0.4, 0.401),
        (5, 5.5),
        (-0.2, 0.2),
        (3, 30),
        (-inf, 0.5),
        (-3, -2.9),
    ]:
        out.append(
            quantile_case(f"truncatedNormal/{lo}/{hi}", "truncatedNormal", stats.truncnorm(lo, hi), lower=lo, upper=hi)
        )
    for alpha in [0.3, 1.0, 7.5]:
        out.append(quantile_case(f"gamma/{alpha}", "gamma", stats.gamma(alpha), alpha=alpha))
    for a, b in [(0.5, 0.5), (2.0, 6.0), (0.1, 0.2), (0.02, 0.03)]:
        out.append(quantile_case(f"beta/{a}/{b}", "beta", stats.beta(a, b), a=a, b=b))
    for kappa in [0.5, 8.0]:
        out.append(quantile_case(f"vonMises/{kappa}", "vonMises", stats.vonmises(kappa), mu=0.0, kappa=kappa))
    alpha = [0.5, 1.0, 2.0]
    for j in [0, 2]:
        marginal = stats.beta(alpha[j], sum(alpha) - alpha[j])
        out.append(quantile_case(f"dirichlet/{j}", "dirichletMarginal", marginal, alpha=alpha, component=j))
    # Rows z L^T with L = [[1, 0], [0.5, 1]] and mean [0, 1]: the second coordinate is N(1, 1.25).
    out.append(
        quantile_case("multivariateNormal/1", "multivariateNormalMarginal", stats.norm(1, math.sqrt(1.25)), component=1)
    )
    return out
