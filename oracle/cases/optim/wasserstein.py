"""Wasserstein distances on the line and sliced, and Wasserstein barycentres, against POT and scipy."""

from typing import cast

import numpy as np
import ot
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.optim_helpers import clouds, histogram
from scipy.stats import wasserstein_distance

REFERENCE: dict[str, str] = {"pot": ot.__version__, "scipy": scipy.__version__}


def as_array(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def line_cases() -> list[Case]:
    g = rng(74)
    u = g.normal(size=7)
    v = g.normal(size=5) * 2 + 1
    uw = g.uniform(0.1, 1.0, size=7)
    vw = g.uniform(0.1, 1.0, size=5)
    out: list[Case] = [
        case(
            "wasserstein1d/p1",
            "wasserstein1d",
            [u, v],
            np.asarray(wasserstein_distance(u, v)),
            rtol=1e-12,
            atol=1e-12,
            p=1,
        ),
        case(
            "wasserstein1d/weighted",
            "wasserstein1d",
            [u, v, uw, vw],
            np.asarray(wasserstein_distance(u, v, uw, vw)),
            rtol=1e-12,
            atol=1e-12,
            p=1,
        ),
    ]
    for p in [2, 3]:
        value = float(cast(float, ot.wasserstein_1d(u, v, uw / uw.sum(), vw / vw.sum(), p=p))) ** (1 / p)
        out.append(
            case(f"wasserstein1d/p{p}", "wasserstein1d", [u, v, uw, vw], np.asarray(value), rtol=1e-11, atol=1e-12, p=p)
        )
    x, y = clouds(75, 8, 6, 3)
    projections = rng(76).normal(size=(3, 10))
    projections /= np.linalg.norm(projections, axis=0, keepdims=True)
    for p in [1, 2]:
        value = float(cast(float, ot.sliced_wasserstein_distance(x, y, projections=projections, p=p)))
        out.append(case(f"sliced/p{p}", "sliced", [x, y, projections], np.asarray(value), rtol=1e-11, atol=1e-12, p=p))
    return out


def barycentre_cases() -> list[Case]:
    n = 12
    grid = np.linspace(0, 1, n)
    cost = (grid[:, None] - grid[None, :]) ** 2
    h1 = np.exp(-((grid - 0.25) ** 2) / 0.01)
    h2 = np.exp(-((grid - 0.75) ** 2) / 0.02)
    a = np.stack([h1 / h1.sum(), h2 / h2.sum()], axis=1)
    weights = np.array([0.3, 0.7])
    # POT's log-domain solver: its scaling solver drifts to a worse point at small reg (higher objective).
    out = [
        case(
            f"barycenter/{reg}",
            "barycenter",
            [a, cost, weights],
            as_array(
                ot.bregman.barycenter(
                    a, cost, reg, weights=weights, method="sinkhorn_log", stopThr=1e-14, numItermax=200000
                )
            ),
            rtol=1e-7,
            atol=1e-11,
            reg=reg,
        )
        for reg in [0.01, 0.05]
    ]
    g = rng(78)
    locations = [g.normal(size=(5, 2)), g.normal(size=(4, 2)) + 2.0, g.normal(size=(6, 2)) - 1.0]
    masses = [histogram(g, 5), histogram(g, 4), histogram(g, 6)]
    init = g.normal(size=(4, 2))
    support = as_array(ot.lp.free_support_barycenter(locations, masses, init, numItermax=200, stopThr=1e-14))
    inputs: list[object] = [init, *locations, *masses]
    out.append(case("freeSupport/0", "freeSupport", inputs, support, rtol=1e-8, atol=1e-10, measures=3))
    return out


def cases() -> list[Case]:
    return line_cases() + barycentre_cases()
