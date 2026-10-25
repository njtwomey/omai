"""Gromov-Wasserstein (conditional gradient and entropic) against POT."""

from typing import cast

import numpy as np
import ot
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.optim_helpers import clouds, histogram

REFERENCE: dict[str, str] = {"pot": ot.__version__}


def as_array(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


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


def cases() -> list[Case]:
    return gromov_cases()
