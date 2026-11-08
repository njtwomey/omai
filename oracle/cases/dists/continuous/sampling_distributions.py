"""Sampling distributions (dists/continuous/sampling-distributions.ts) against scipy.stats."""

import numpy as np
import scipy
from omai_oracle.distributions import distribution_cases, method_case
from omai_oracle.fixture import Case
from omai_oracle.helpers import rng
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(37)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    pos = np.exp(g.normal(size=8))
    x = g.normal(size=8) * 2
    out: list[Case] = []
    for i, (d1, d2) in enumerate([(3.0, 6.0), (10.0, 20.0), (1.5, 3.5)]):
        p = [np.array([d1]), np.array([d2])]
        out += distribution_cases(
            f"fDistribution/{i}",
            "fDistribution",
            p,
            stats.f(*p),
            pos,
            q,
            skip=("logCdf", "logSf") + (("variance",) if d2 <= 4 else ()),
        )
    for i, (df, nc) in enumerate([(3.0, 2.0), (1.0, 0.5), (8.0, 25.0)]):
        p = [np.array([df]), np.array([nc])]
        out += distribution_cases(
            f"noncentralChi2/{i}",
            "noncentralChi2",
            p,
            stats.ncx2(*p),
            pos * (1 + i * 4),
            q,
            skip=("logCdf", "logSf"),
            # Boost's series and scipy's numerical entropy and quantiles carry about these accuracies.
            tolerances={"logProb": 1e-9, "cdf": 1e-9, "sf": 1e-9, "ppf": 1e-8, "isf": 1e-8, "entropy": 1e-7},
        )
    for i, (df, nc) in enumerate([(5.0, 1.0), (12.0, -0.8), (3.5, 0.0)]):
        p = [np.array([df]), np.array([nc])]
        out += distribution_cases(
            f"noncentralT/{i}",
            "noncentralT",
            p,
            stats.nct(*p),
            x,
            q,
            skip=("logCdf", "logSf"),
            tolerances={"logProb": 1e-8, "cdf": 1e-8, "sf": 1e-8, "ppf": 1e-8, "isf": 1e-8, "entropy": 1e-7},
        )
    for i, (d1, d2, nc) in enumerate([(4.0, 10.0, 2.0), (2.0, 30.0, 8.0), (6.5, 5.5, 0.5)]):
        p = [np.array([d1]), np.array([d2]), np.array([nc])]
        out += distribution_cases(
            f"noncentralF/{i}",
            "noncentralF",
            p,
            stats.ncf(*p),
            pos * (1 + i),
            q,
            skip=("logCdf", "logSf") + (("variance",) if d2 <= 4 else ()),
            tolerances={"logProb": 1e-9, "cdf": 1e-9, "sf": 1e-9, "ppf": 1e-8, "isf": 1e-8, "entropy": 1e-7},
        )
    # The studentised range is a double integral in both libraries (scipy's adaptive, ours a fixed product rule).
    ranges = np.array([0.5, 1.0, 2.0, 3.5, 5.0])
    for i, (k, df) in enumerate([(3.0, 10.0), (5.0, np.inf)]):
        p = [np.array([k]), np.array([df])]
        sr = stats.studentized_range(*p)
        label = f"studentisedRange/{i}"
        out += [
            method_case(label, "studentisedRange", "logProb", p, ranges, sr.logpdf(ranges), rtol=1e-7),
            method_case(label, "studentisedRange", "cdf", p, ranges, sr.cdf(ranges), rtol=1e-7),
            method_case(label, "studentisedRange", "ppf", p, q[[0, 3]], sr.ppf(q[[0, 3]]), rtol=1e-7),
        ]
        # scipy's moments are NaN for infinite df.
        if np.isfinite(df):
            out += [
                method_case(label, "studentisedRange", "mean", p, None, sr.mean(), rtol=1e-7),
                method_case(label, "studentisedRange", "variance", p, None, sr.var(), rtol=1e-7),
            ]
    return out
