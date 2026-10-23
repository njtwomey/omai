"""Elliptic integrals and Jacobi elliptic functions against mpmath."""

from functools import partial

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def jacobi(part: str, u: Mpf, m: Mpf) -> Mpf:
    return mp.ellipfun(part, u, m=m)


def cases() -> list[Case]:
    g = rng(9)
    m = np.concatenate([g.uniform(-10, 1, 25), 1 - np.exp(g.uniform(np.log(1e-12), np.log(1e-2), 5)), [0.0]])
    out: list[Case] = []
    out += value_and_grads("ellipk", "ellipk", mp.ellipk, [m], wrt=[0], rtol=1e-14, grad_rtol=1e-11)
    out += value_and_grads("ellipe", "ellipe", mp.ellipe, [m], wrt=[0], rtol=1e-14, grad_rtol=1e-11)
    phi = np.concatenate([g.uniform(-1.5, 1.5, 20), g.uniform(-12, 12, 10)])
    mm = np.concatenate([g.uniform(-5, 0.99, 29), [0.0]])
    out += value_and_grads(
        "ellipkinc",
        "ellipkinc",
        mp.ellipf,
        [phi, mm],
        wrt=[0, 1],
        rtol=1e-14,
        atol=1e-15,
        grad_rtol=1e-11,
        grad_atol=1e-14,
    )
    out += value_and_grads(
        "ellipeinc",
        "ellipeinc",
        mp.ellipe,
        [phi, mm],
        wrt=[0, 1],
        rtol=1e-14,
        atol=1e-15,
        grad_rtol=1e-11,
        grad_atol=1e-14,
    )
    u = g.uniform(-10, 10, 30)
    jm = g.uniform(0, 1, 30)
    for part in ["sn", "cn", "dn"]:
        out += value_and_grads(
            f"ellipj/{part}",
            "ellipj",
            partial(jacobi, part),
            [u, jm],
            wrt=[0],
            rtol=1e-13,
            atol=1e-14,
            grad_atol=1e-13,
            part=part,
        )
    return out
