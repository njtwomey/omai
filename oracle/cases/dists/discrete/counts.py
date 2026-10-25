"""Count distributions (dists/discrete/counts.ts) against scipy.stats and statsmodels."""

import numpy as np
import scipy
import statsmodels
from mdml_oracle.distributions import distribution_cases
from mdml_oracle.fixture import Case
from scipy import stats
from statsmodels.distributions.discrete import zinegbin, zipoisson

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    q = np.array([0.05, 0.3, 0.5, 0.77, 0.95])
    counts = np.array([0.0, 1, 2, 3, 5, 8, 13])
    out: list[Case] = []
    for i, lam in enumerate([0.7, 4.0, 30.0]):
        ll = np.array([lam])
        out += distribution_cases(
            f"poisson/{i}", "poisson", [ll], stats.poisson(ll), counts * (1 + i), q, discrete=True, skip=("isf",)
        )
    for i, p in enumerate([0.2, 0.7]):
        pp = np.array([p])
        out += distribution_cases(
            f"geometric/{i}", "geometric", [pp], stats.geom(pp), counts + 1, q, discrete=True, skip=("isf",)
        )
    for i, (r, p) in enumerate([(3.0, 0.4), (0.7, 0.2)]):
        rr, pp = np.array([r]), np.array([p])
        out += distribution_cases(
            f"negativeBinomial/{i}",
            "negativeBinomial",
            [rr, pp],
            stats.nbinom(rr, pp),
            counts,
            q,
            discrete=True,
            skip=("isf",),
        )
    for i, (m, n, draws) in enumerate([(20, 7, 12), (50, 30, 5)]):
        p = [np.array([float(m)]), np.array([float(n)]), np.array([float(draws)])]
        out += distribution_cases(
            f"hypergeometric/{i}", "hypergeometric", p, stats.hypergeom(*p), counts, q, discrete=True, skip=("isf",)
        )
    for i, (lam, w) in enumerate([(2.0, 0.3), (6.0, 0.1)]):
        p = [np.array([lam]), np.array([w])]
        out += distribution_cases(
            f"zeroInflatedPoisson/{i}",
            "zeroInflatedPoisson",
            p,
            zipoisson(*p),
            counts,
            q,
            discrete=True,
            # statsmodels' generic ppf misplaces the support start (-1 where P(0) covers q), and its sf is 1 - cdf.
            skip=("isf", "logCdf", "logSf", "ppf"),
            tolerances={"sf": 1e-7},
        )
    # statsmodels' NB2 parameterisation: size 1 / alpha and mean mu, i.e. prob = size / (size + mu).
    for i, (size, prob, w) in enumerate([(3.0, 0.4, 0.25), (1.5, 0.6, 0.5)]):
        mu = size * (1 - prob) / prob
        p = [np.array([size]), np.array([prob]), np.array([w])]
        frozen = zinegbin(np.array([mu]), np.array([1 / size]), 2.0, np.array([w]))
        out += distribution_cases(
            f"zeroInflatedNegativeBinomial/{i}",
            "zeroInflatedNegativeBinomial",
            p,
            frozen,
            counts,
            q,
            discrete=True,
            skip=("isf", "logCdf", "logSf", "entropy", "ppf"),
            tolerances={"sf": 1e-7},
        )
    return out
