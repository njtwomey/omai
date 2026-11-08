"""LQR gains against `scipy.linalg.solve_discrete_are`: `P` and `K = (R + B^T P B)^{-1} B^T P A`.

Systems: random stabilisable ones of a few sizes and the zero-order-hold double integrator (`cont2discrete`).
"""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd
from scipy.linalg import solve, solve_discrete_are
from scipy.signal import cont2discrete

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def lqr_cases(out: list[Case], name: str, a: FloatArray, b: FloatArray, q: FloatArray, r: FloatArray) -> None:
    """The cost-to-go and gain cases of one problem (direct linear algebra: 1e-10)."""
    p = solve_discrete_are(a, b, q, r)
    k = solve(r + b.T @ p @ b, b.T @ p @ a)
    out.append(case(f"{name}/P", "lqrP", [a, b, q, r], p, rtol=1e-10, atol=1e-10))
    out.append(case(f"{name}/K", "lqrK", [a, b, q, r], k, rtol=1e-10, atol=1e-10))


def cases() -> list[Case]:
    g = rng(72)
    out: list[Case] = []
    for n, m in [(2, 1), (3, 1), (4, 2)]:
        a = g.normal(size=(n, n)) * 0.6
        b = g.normal(size=(n, m))
        lqr_cases(out, f"random/{n}x{m}", a, b, spd(g, n), spd(g, m))
    for dt, r in [(0.1, 1.0), (0.05, 0.1)]:
        ad, bd, *_ = cont2discrete(
            (np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), np.eye(2), np.zeros((2, 1))),
            dt,
            method="zoh",
        )
        lqr_cases(out, f"double-integrator/{dt}/{r}", ad, bd, np.eye(2), np.array([[r]]))
    return out
