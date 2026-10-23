"""Optimal transport against POT (exact, entropic, unbalanced, sliced, Gromov-Wasserstein, barycentres) and scipy."""

from typing import cast

import numpy as np
import ot
import scipy
from scipy.stats import wasserstein_distance
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"pot": ot.__version__, "scipy": scipy.__version__}


def histogram(g: np.random.Generator, n: int) -> FloatArray:
    h = g.uniform(0.2, 1.0, size=n)
    return h / h.sum()


def clouds(seed: int, n: int, m: int, d: int) -> tuple[FloatArray, FloatArray]:
    g = rng(seed)
    return g.normal(size=(n, d)), g.normal(size=(m, d)) + 1.0


def as_array(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def exact_cases() -> list[Case]:
    out: list[Case] = []
    for i, (n, m) in enumerate([(5, 7), (8, 8)]):
        g = rng(70 + i)
        x, y = clouds(70 + i, n, m, 2)
        a, b = histogram(g, n), histogram(g, m)
        cost = as_array(ot.dist(x, y))
        plan = as_array(ot.emd(a, b, cost))
        out.append(case(f"emd/{i}", "emd", [a, b, cost], {"plan": plan, "cost": float(np.sum(plan * cost))}))
    # Uniform weights on equal sizes: a permutation (degenerate pivots throughout).
    x, y = clouds(72, 6, 6, 2)
    u = np.full(6, 1 / 6)
    cost = as_array(ot.dist(x, y))
    plan = as_array(ot.emd(u, u, cost))
    out.append(case("emd/permutation", "emd", [u, u, cost], {"plan": plan, "cost": float(np.sum(plan * cost))}))
    return [c.model_copy(update={"rtol": 1e-10, "atol": 1e-12}) for c in out]


def entropic_cases() -> list[Case]:
    out: list[Case] = []
    g = rng(73)
    x, y = clouds(73, 6, 5, 2)
    a, b = histogram(g, 6), histogram(g, 5)
    cost = as_array(ot.dist(x, y))
    cost /= cost.max()
    for reg in [0.05, 0.5]:
        plan = as_array(ot.sinkhorn(a, b, cost, reg, method="sinkhorn_log", stopThr=1e-14, numItermax=200000))
        out.append(case(f"sinkhorn/{reg}", "sinkhorn", [a, b, cost], plan, rtol=1e-8, atol=1e-11, reg=reg))
    plan = as_array(ot.sinkhorn(a, b, cost, 0.5, method="sinkhorn", stopThr=1e-14, numItermax=200000))
    out.append(
        case("sinkhorn/standard", "sinkhorn", [a, b, cost], plan, rtol=1e-8, atol=1e-11, reg=0.5, method="standard")
    )
    # Unbalanced: different masses, KL relaxation.
    a2 = a * 2.0
    for reg, reg_m in [(0.1, 1.0), (0.5, 0.3)]:
        plan = as_array(
            ot.unbalanced.sinkhorn_unbalanced(
                a2, b, cost, reg, reg_m, method="sinkhorn", stopThr=1e-14, numItermax=200000
            )
        )
        out.append(
            case(
                f"unbalanced/{reg}-{reg_m}",
                "unbalanced",
                [a2, b, cost],
                plan,
                rtol=1e-7,
                atol=1e-11,
                reg=reg,
                regM=reg_m,
            )
        )
    return out


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


def gromov_cases() -> list[Case]:
    out: list[Case] = []
    g = rng(77)
    x, y = clouds(77, 6, 5, 2)
    c1 = as_array(ot.dist(x, x))
    c2 = as_array(ot.dist(y, y))
    c1 /= c1.max()
    c2 /= c2.max()
    p, q = histogram(g, 6), histogram(g, 5)
    plan, log = ot.gromov.gromov_wasserstein(c1, c2, p, q, "square_loss", log=True, tol_rel=1e-12, tol_abs=1e-14)
    out.append(
        case(
            "gromov/square",
            "gromov",
            [c1, c2, p, q],
            {"plan": as_array(plan), "distance": float(cast(float, log["gw_dist"]))},
            rtol=1e-8,
            atol=1e-10,
        )
    )
    plan, log = ot.gromov.entropic_gromov_wasserstein(
        c1,
        c2,
        p,
        q,
        "square_loss",
        epsilon=0.05,
        log=True,
        tol=1e-13,
        max_iter=100000,
        stopThr=1e-14,
        numItermax=200000,
    )
    out.append(
        case(
            "entropicGromov/square",
            "entropicGromov",
            [c1, c2, p, q],
            {"plan": as_array(plan), "distance": float(cast(float, log["gw_dist"]))},
            rtol=1e-7,
            atol=1e-10,
            epsilon=0.05,
        )
    )
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
    return exact_cases() + entropic_cases() + line_cases() + gromov_cases() + barycentre_cases()
