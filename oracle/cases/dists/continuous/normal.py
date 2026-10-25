"""Normal-family distributions (dists/continuous/normal.ts) against scipy.stats."""

import numpy as np
import scipy
from mdml_oracle.distributions import distribution_cases
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import rng
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(30)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    x = g.normal(size=8)
    pos = np.exp(g.normal(size=8))
    loc, scale = np.array([0.5, -1.0, 2.0]), np.array([1.0, 0.5, 3.0])
    out: list[Case] = []
    for i in range(3):
        a, s = loc[i : i + 1], scale[i : i + 1]
        out += distribution_cases(f"normal/{i}", "normal", [a, s], stats.norm(a, s), x, q)
        out += distribution_cases(f"logNormal/{i}", "logNormal", [a, s], stats.lognorm(s, scale=np.exp(a)), pos, q)
        out += distribution_cases(f"halfNormal/{i}", "halfNormal", [s], stats.halfnorm(scale=s), np.abs(x), q)
    # A batch of normals against a column of points.
    out += distribution_cases(
        "normal/batch", "normal", [loc, scale], stats.norm(loc, scale), x[:3], q[:3], skip=("logCdf", "logSf")
    )
    for i, (lo_, hi_) in enumerate([(-1.0, 1.5), (0.8, 4.0), (-np.inf, 0.0)]):
        lo, hi = np.array([lo_]), np.array([hi_])
        frozen = stats.truncnorm((lo - loc[:1]) / scale[:1], (hi - loc[:1]) / scale[:1], loc[:1], scale[:1])
        pts = np.clip(x, max(lo_, -3.0), hi_)
        out += distribution_cases(
            f"truncatedNormal/{i}",
            "truncatedNormal",
            [loc[:1], scale[:1], lo, hi],
            frozen,
            pts,
            q,
            rtol=1e-9,
            # scipy's entropy is NaN for an infinite bound (it evaluates -inf * 0).
            skip=("isf", "entropy") if np.isinf(lo_) else ("isf",),
        )
    for i, (a_, l_, s_) in enumerate([(0.0, 0.0, 1.0), (3.0, 0.5, 2.0), (-1.5, -1.0, 0.7)]):
        a, lc, sc = np.array([a_]), np.array([l_]), np.array([s_])
        out += distribution_cases(
            f"skewNormal/{i}",
            "skewNormal",
            [lc, sc, a],
            stats.skewnorm(a, lc, sc),
            x * 2,
            q,
            # scipy integrates the thin tail numerically (quad, epsrel 1.5e-8), so its tail values carry that accuracy.
            tolerances={"ppf": 1e-9, "isf": 1e-9, "entropy": 1e-9, "cdf": 1e-8, "sf": 1e-8},
            skip=("logCdf", "logSf"),
        )
    return out
