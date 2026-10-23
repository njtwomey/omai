"""Gamma-family distributions (dists/continuous/gamma.ts) against scipy.stats."""

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.distributions import distribution_cases
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(34)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    pos = np.exp(g.normal(size=8))
    out: list[Case] = []
    for i, (alpha, rate) in enumerate([(0.5, 1.0), (2.0, 4.0), (30.0, 0.5)]):
        a, r = np.array([alpha]), np.array([rate])
        out += distribution_cases(f"gamma/{i}", "gamma", [a, r], stats.gamma(a, scale=1 / r), pos, q)
        out += distribution_cases(f"exponential/{i}", "exponential", [r], stats.expon(scale=1 / r), pos, q)
    # A batch of gammas evaluated at a column of points.
    alphas, rates = np.array([0.7, 3.0]), np.array([2.0, 0.5])
    out += distribution_cases(
        "gamma/batch", "gamma", [alphas, rates], stats.gamma(alphas, scale=1 / rates), pos[:2], q[:2]
    )
    for i, (k, rate) in enumerate([(1.0, 2.0), (4.0, 0.5)]):
        kk, r = np.array([k]), np.array([rate])
        out += distribution_cases(f"erlang/{i}", "erlang", [kk, r], stats.erlang(kk, scale=1 / r), pos, q)
    for i, (alpha, beta) in enumerate([(3.5, 2.0), (1.5, 0.5)]):
        ia, ib = np.array([alpha]), np.array([beta])
        out += distribution_cases(
            f"inverseGamma/{i}",
            "inverseGamma",
            [ia, ib],
            stats.invgamma(ia, scale=ib),
            pos,
            q,
            skip=("variance",) if alpha <= 2 else (),
        )
    for i, k in enumerate([1.0, 5.0]):
        kk = np.array([k])
        out += distribution_cases(f"chi2/{i}", "chi2", [kk], stats.chi2(kk), pos * 3, q)
    for i, (k, lam) in enumerate([(1.7, 2.5), (0.6, 1.0)]):
        wk, wl = np.array([k]), np.array([lam])
        out += distribution_cases(f"weibull/{i}", "weibull", [wk, wl], stats.weibull_min(wk, scale=wl), pos, q)
    for i, (mu, lam) in enumerate([(1.0, 1.0), (2.0, 4.0), (0.5, 20.0)]):
        m, lm = np.array([mu]), np.array([lam])
        out += distribution_cases(
            f"inverseGaussian/{i}",
            "inverseGaussian",
            [m, lm],
            stats.invgauss(m / lm, scale=lm),
            pos,
            q,
            tolerances={"ppf": 1e-9, "isf": 1e-9, "entropy": 1e-9},
        )
    return out
