"""Heavy-tailed distributions (dists/continuous/heavy-tailed.ts) against scipy.stats."""

import numpy as np
import scipy
from omai_oracle.distributions import distribution_cases
from omai_oracle.fixture import Case
from omai_oracle.helpers import rng
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(33)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    x = g.normal(size=8) * 3
    pos = np.exp(g.normal(size=8))
    out: list[Case] = []
    for i, (df, loc, scale) in enumerate([(2.5, 1.0, 2.0), (5.0, 0.0, 1.0), (30.0, -1.0, 0.5)]):
        p = [np.array([df]), np.array([loc]), np.array([scale])]
        out += distribution_cases(f"studentT/{i}", "studentT", p, stats.t(*p), x, q, skip=("logCdf", "logSf"))
    c_loc, c_scale = np.array([0.5]), np.array([2.0])
    out += distribution_cases(
        "cauchy",
        "cauchy",
        [c_loc, c_scale],
        stats.cauchy(c_loc, c_scale),
        x * 2,
        q,
        skip=("logCdf", "logSf", "mean", "variance"),
    )
    out += distribution_cases(
        "halfCauchy",
        "halfCauchy",
        [c_scale],
        stats.halfcauchy(scale=c_scale),
        pos * 3,
        q,
        skip=("logCdf", "logSf", "mean", "variance"),
    )
    out += distribution_cases("laplace", "laplace", [c_loc, c_scale], stats.laplace(c_loc, c_scale), x, q)
    out += distribution_cases("logistic", "logistic", [c_loc, c_scale], stats.logistic(c_loc, c_scale), x, q)
    for i, (b, xm) in enumerate([(3.5, 1.5), (1.5, 0.5)]):
        pb, pm = np.array([b]), np.array([xm])
        out += distribution_cases(
            f"pareto/{i}", "pareto", [pb, pm], stats.pareto(pb, scale=pm), pos + xm, q, skip=("logCdf",)
        )
    for i, (a, b, loc, scale) in enumerate([(0.0, 1.0, 0.0, 1.0), (1.2, 2.0, 0.5, 1.5), (-0.7, 0.8, -1.0, 0.6)]):
        p = [np.array([a]), np.array([b]), np.array([loc]), np.array([scale])]
        out += distribution_cases(
            f"johnsonSU/{i}",
            "johnsonSU",
            p,
            stats.johnsonsu(*p),
            x,
            q,
            # scipy integrates the entropy numerically.
            tolerances={"entropy": 1e-8},
        )
    return out
