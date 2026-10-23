"""Elliptic integrals and Jacobi elliptic functions against scipy.special."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.special as sp
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import FloatArray, reference, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def jacobi(part: int) -> Callable[[FloatArray, FloatArray], FloatArray]:
    def f(u: FloatArray, m: FloatArray) -> FloatArray:
        return sp.ellipj(u, m)[part]

    return f


def cases() -> list[Case]:
    g = rng(9)
    m = np.concatenate([g.uniform(-5, 0.99, 25), [0.0, 0.5]])
    phi, mm = g.uniform(-6, 6, 25), g.uniform(-3, 0.95, 25)
    u, jm = g.uniform(-6, 6, 25), g.uniform(0, 1, 25)
    return [
        reference("ellipk", "ellipk", sp.ellipk, [m]),
        reference("ellipe", "ellipe", sp.ellipe, [m]),
        reference("ellipkinc", "ellipkinc", sp.ellipkinc, [phi, mm]),
        reference("ellipeinc", "ellipeinc", sp.ellipeinc, [phi, mm]),
        reference("ellipj/sn", "ellipj", jacobi(0), [u, jm], atol=1e-14, part="sn"),
        reference("ellipj/cn", "ellipj", jacobi(1), [u, jm], atol=1e-14, part="cn"),
        reference("ellipj/dn", "ellipj", jacobi(2), [u, jm], atol=1e-14, part="dn"),
    ]
