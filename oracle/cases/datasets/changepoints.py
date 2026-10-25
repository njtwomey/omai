"""Exact BOCPD evidence against brute force over every segmentation of short series.

Each segmentation's prior is h^(changes) (1 - h)^(n - 1 - changes); a segment's marginal likelihood comes from
scipy.stats.multivariate_normal (Gaussian mean and trend models) or scipy.integrate.quad over the parameter
(variance, rate and AR models).
"""

import itertools
import math

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray
from scipy import integrate, stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}
HAZARD = 0.2


def segment_log_marginal(kind: str, y: FloatArray, prev: float) -> float:
    n = len(y)
    if kind == "mean":
        sigma, prior_std = 0.7, 2.0
        cov = sigma**2 * np.eye(n) + prior_std**2 * np.ones((n, n))
        return float(stats.multivariate_normal(np.zeros(n), cov).logpdf(y))
    if kind == "trend":
        sigma, a_std, b_std = 0.5, 2.0, 0.3
        x = np.stack([np.ones(n), np.arange(n, dtype=np.float64)], axis=1)
        cov = sigma**2 * np.eye(n) + x @ np.diag([a_std**2, b_std**2]) @ x.T
        return float(stats.multivariate_normal(np.zeros(n), cov).logpdf(y))
    if kind == "variance":
        alpha, beta = 3.0, 2.0

        def integrand_var(v: float) -> float:
            prior = stats.invgamma(alpha, scale=beta).pdf(v)
            return float(prior * np.prod(stats.norm(0, math.sqrt(v)).pdf(y)))

        return math.log(integrate.quad(integrand_var, 0, np.inf, limit=400)[0])
    if kind == "rate":
        shape, rate = 2.0, 0.5

        def integrand_rate(lam: float) -> float:
            return float(stats.gamma(shape, scale=1 / rate).pdf(lam) * np.prod(stats.poisson(lam).pmf(y)))

        # A finite range with breakpoints: quad over [0, inf) loses accuracy on the peaked integrand.
        return math.log(integrate.quad(integrand_rate, 0, 200, limit=400, points=[1, 2, 5, 10, 20, 50])[0])
    sigma, prior_std = 0.6, 0.5
    lags = np.concatenate([[prev], y[:-1]])

    def integrand_ar(phi: float) -> float:
        return float(stats.norm(0, prior_std).pdf(phi) * np.prod(stats.norm(phi * lags, sigma).pdf(y)))

    return math.log(integrate.quad(integrand_ar, -10, 10, limit=200)[0])


def brute_force_evidence(kind: str, y: FloatArray, min_segment: int = 1) -> float:
    n = len(y)
    total = 0.0
    for cuts in itertools.product([0, 1], repeat=n - 1):
        starts = [0] + [i + 1 for i, c in enumerate(cuts) if c]
        bounds = [*starts, n]
        # Before step t the current run has t - start observations; cuts need at least min_segment of them.
        log_p = 0.0
        start = 0
        feasible = True
        for t in range(1, n):
            run = t - start
            if run < min_segment:
                if cuts[t - 1]:
                    feasible = False
                    break
                continue
            log_p += math.log(HAZARD) if cuts[t - 1] else math.log(1 - HAZARD)
            if cuts[t - 1]:
                start = t
        if not feasible:
            continue
        for a, b in itertools.pairwise(bounds):
            prev = float(y[a - 1]) if a > 0 else 0.0
            log_p += segment_log_marginal(kind, y[a:b], prev)
        total += math.exp(log_p)
    return math.log(total)


def cases() -> list[Case]:
    out: list[Case] = []
    series = {
        "mean": np.asarray([0.1, -0.4, 0.3, 2.9, 3.4, 3.1]),
        "trend": np.asarray([0.2, 0.4, 0.9, 3.0, 2.8, 2.5]),
        "variance": np.asarray([0.3, -0.5, 0.2, 3.1, -2.7, 2.2]),
        "rate": np.asarray([2.0, 1.0, 3.0, 9.0, 11.0, 8.0]),
        "ar": np.asarray([0.5, 0.3, 0.2, -1.0, 0.9, -0.8]),
    }
    for kind, y in series.items():
        out.append(
            case(f"evidence/{kind}", "exactBocpd", [y], np.asarray(brute_force_evidence(kind, y)), rtol=1e-7, kind=kind)
        )
    y = series["mean"]
    out.append(
        case(
            "evidence/mean/min-segment-2",
            "exactBocpd",
            [y],
            np.asarray(brute_force_evidence("mean", y, 2)),
            rtol=1e-7,
            kind="mean",
            minSegment=2,
        )
    )
    return out
