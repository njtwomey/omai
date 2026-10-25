"""Hankel singular values and balanced truncation (truncate and matchdc, stable and unstable) against python-control."""

from typing import Literal, cast

import control as ct
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.lti import parts, stable_system

REFERENCE: dict[str, str] = {"control": ct.__version__}

W: FloatArray = np.logspace(-1, 1, 5)


def response(sys: ct.StateSpace) -> FloatArray:
    """The frequency response at W as `[2, nw * outputs * inputs]` parts, frequency first."""
    h = np.asarray(sys(1j * W))
    return parts(h) if h.ndim == 1 else parts(np.moveaxis(h, -1, 0).ravel())


def cases() -> list[Case]:
    g = rng(4167)
    out: list[Case] = []
    a, b, c, d = stable_system(g, 5, 2, 2)
    sys = ct.ss(a, b, c, d)
    out.append(case("hsvd", "hsvd", [a, b, c, d], np.real(np.asarray(ct.hsvd(sys))), rtol=1e-9, atol=1e-11))
    methods: list[Literal["truncate", "matchdc"]] = ["truncate", "matchdc"]
    for method in methods:
        for order in [2, 3]:
            reduced = cast(ct.StateSpace, ct.balred(sys, order, method=method))
            out.append(
                case(
                    f"balred/{method}/{order}",
                    "balred",
                    [a, b, c, d],
                    response(reduced),
                    rtol=1e-8,
                    atol=1e-10,
                    order=order,
                    method=method,
                )
            )
    # An unstable mode is kept and the stable part reduced.
    ua = np.diag([-1.0, -3.0, -8.0, 0.5])
    ua[0, 1] = 0.7
    ub = np.array([[1.0], [0.5], [1.0], [1.0]])
    uc = np.array([[1.0, -1.0, 0.4, 1.0]])
    usys = ct.ss(ua, ub, uc, np.zeros((1, 1)))
    reduced = cast(ct.StateSpace, ct.balred(usys, 3))
    out.append(
        case(
            "balred/unstable",
            "balred",
            [ua, ub, uc, np.zeros((1, 1))],
            response(reduced),
            rtol=1e-8,
            atol=1e-10,
            order=3,
            method="truncate",
        )
    )
    return out
