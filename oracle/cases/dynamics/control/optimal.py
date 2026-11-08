"""Optimal control and estimation against python-control: lqr, dlqr (with cross terms), lqe, dlqe, and the LQG
controller assembled from python-control's lqr and lqe gains (python-control 0.10 has no lqg function).

python-control requires G QN G^T to be exactly symmetric, so the noise enters through G = I.
"""

import control as ct
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.lti import schur_stable_system, stable_system

REFERENCE: dict[str, str] = {"control": ct.__version__}


def weights(g: np.random.Generator, n: int) -> FloatArray:
    m = g.normal(size=(n, n))
    w = m @ m.T + 0.5 * np.eye(n)
    return 0.5 * (w + w.T)


def cases() -> list[Case]:
    g = rng(26_1)
    out: list[Case] = []
    a, b, c, _d = stable_system(g, 4, 2, 2)
    q = weights(g, 4)
    r = weights(g, 2)
    n = 0.1 * g.normal(size=(4, 2))
    k, s, _e = ct.lqr(a, b, q, r)
    out.append(case("lqr/K", "lqr_K", [a, b, q, r], k, rtol=1e-8, atol=1e-10))
    out.append(case("lqr/S", "lqr_S", [a, b, q, r], s, rtol=1e-8, atol=1e-10))
    k, s, _e = ct.lqr(a, b, q, r, n)
    out.append(case("lqr/cross", "lqr_K", [a, b, q, r, n], k, rtol=1e-8, atol=1e-10))
    da, db, dc, _dd = schur_stable_system(g, 3, 2, 2)
    dq = weights(g, 3)
    dr = weights(g, 2)
    dn = 0.1 * g.normal(size=(3, 2))
    k, s, _e = ct.dlqr(da, db, dq, dr)
    out.append(case("dlqr/K", "dlqr_K", [da, db, dq, dr], k, rtol=1e-8, atol=1e-10))
    out.append(case("dlqr/S", "dlqr_S", [da, db, dq, dr], s, rtol=1e-8, atol=1e-10))
    k, s, _e = ct.dlqr(da, db, dq, dr, dn)
    out.append(case("dlqr/cross", "dlqr_K", [da, db, dq, dr, dn], k, rtol=1e-8, atol=1e-10))
    gn = np.eye(4)
    qn = weights(g, 4)
    rn = weights(g, 2)
    el, p, _e = ct.lqe(a, gn, c, qn, rn)
    out.append(case("lqe/L", "lqe_L", [a, gn, c, qn, rn], el, rtol=1e-8, atol=1e-10))
    out.append(case("lqe/P", "lqe_P", [a, gn, c, qn, rn], p, rtol=1e-8, atol=1e-10))
    dg = np.eye(3)
    dqn = weights(g, 3)
    el, p, _e = ct.dlqe(da, dg, dc, dqn, rn)
    out.append(case("dlqe/L", "dlqe_L", [da, dg, dc, dqn, rn], el, rtol=1e-8, atol=1e-10))
    out.append(case("dlqe/P", "dlqe_P", [da, dg, dc, dqn, rn], p, rtol=1e-8, atol=1e-10))
    kk, _s, _e = ct.lqr(a, b, q, r)
    ll, _p, _e = ct.lqe(a, b, c, np.eye(2), rn)
    ac = a - b @ kk - ll @ c
    out.append(case("lqg/A", "lqg_A", [a, b, c, q, r, rn], ac, rtol=1e-8, atol=1e-10))
    out.append(case("lqg/B", "lqg_B", [a, b, c, q, r, rn], ll, rtol=1e-8, atol=1e-10))
    out.append(case("lqg/C", "lqg_C", [a, b, c, q, r, rn], -kk, rtol=1e-8, atol=1e-10))
    return out
