"""Extreme-value distributions (dists/continuous/extreme.ts) against scipy.stats."""

import numpy as np
import scipy
from mdml_oracle.distributions import distribution_cases
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import rng
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(36)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    x = g.normal(size=8)
    pos = np.exp(g.normal(size=8))
    out: list[Case] = []
    loc, scale = np.array([0.5]), np.array([2.0])
    out += distribution_cases("gumbel", "gumbel", [loc, scale], stats.gumbel_r(loc, scale), x * 3, q)
    # The extreme-value shape xi is scipy's -c.
    for i, xi in enumerate([0.0, 0.3, -0.4]):
        p = [loc, scale, np.array([xi])]
        frozen = stats.genextreme(-p[2], loc, scale)
        pts = np.clip(x, -2.0, 4.0)
        out += distribution_cases(
            f"generalisedExtremeValue/{i}", "generalisedExtremeValue", p, frozen, pts, q, skip=("logSf",)
        )
    for i, xi in enumerate([0.0, 0.25, -0.3]):
        p = [loc, scale, np.array([xi])]
        frozen = stats.genpareto(p[2], loc, scale)
        out += distribution_cases(
            f"generalisedPareto/{i}", "generalisedPareto", p, frozen, loc + pos * 1.5, q, skip=("logCdf",)
        )
    for i, alpha in enumerate([1.5, 3.0]):
        p = [np.array([alpha]), np.array([-0.5]), np.array([1.5])]
        frozen = stats.invweibull(*p)
        out += distribution_cases(
            f"frechet/{i}", "frechet", p, frozen, pos, q, skip=("logSf",) + (("variance",) if alpha <= 2 else ())
        )
    return out
