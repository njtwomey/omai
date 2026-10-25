"""Second-order cone and semidefinite programs against cvxpy (Clarabel): optimal points and values."""

from importlib.metadata import version
from typing import cast

import cvxpy as cp
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy"), "clarabel": version("clarabel")}

SOLVE: dict[str, float] = {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12}


def socp_case(name: str, seed: int, linear: bool) -> Case:
    """cvxpy's SOCP example: min f^T x s.t. |A_i x + b_i| <= c_i^T x + d_i, F x = g (optionally G x <= h)."""
    g = rng(seed)
    m, n, p, k = 3, 5, 1, 3
    x0 = g.normal(size=n)
    f = g.normal(size=n)
    a = [g.normal(size=(k, n)) for _ in range(m)]
    b = [g.normal(size=k) for _ in range(m)]
    c = [g.normal(size=n) for _ in range(m)]
    d = [float(np.linalg.norm(a[i] @ x0 + b[i]) - c[i] @ x0 + 0.5) for i in range(m)]
    F = g.normal(size=(p, n))
    gg = F @ x0
    G = g.normal(size=(2, n)) if linear else np.zeros((0, n))
    h = G @ x0 + 0.3 if linear else np.zeros(0)
    x = cp.Variable(n)
    constraints: list[cp.Constraint] = [cp.SOC(c[i] @ x + d[i], a[i] @ x + b[i]) for i in range(m)]
    constraints.append(F @ x == gg)
    if linear:
        constraints.append(G @ x <= h)
    problem = cp.Problem(cp.Minimize(f @ x), constraints)
    problem.solve(solver="CLARABEL", **SOLVE)
    inputs: list[object] = [f, np.stack(a), np.stack(b), np.stack(c), np.array(d), F, gg, G, h]
    return case(name, "socp", inputs, {"x": np.asarray(x.value), "fun": np.array(problem.value)}, rtol=1e-6, atol=1e-6)


def sdp_case(name: str, seed: int, p: int, m: int) -> Case:
    """min tr(C X) s.t. tr(A_i X) = b_i, X PSD, with C positive definite and b from a positive definite X0."""
    g = rng(seed)
    M = g.normal(size=(p, p))
    C = M @ M.T + 0.1 * np.eye(p)
    W = g.normal(size=(p, p))
    X0 = W @ W.T / p + np.eye(p)
    a: list[FloatArray] = []
    b: list[float] = []
    for _ in range(m):
        ai = g.normal(size=(p, p))
        ai = ai + ai.T
        a.append(ai)
        b.append(float(np.trace(ai @ X0)))
    X = cp.Variable((p, p), symmetric=True)
    constraints: list[cp.Constraint] = [cast(cp.Constraint, X >> 0)]
    constraints += [cast(cp.Constraint, cp.trace(a[i] @ X) == b[i]) for i in range(m)]
    problem = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    problem.solve(solver="CLARABEL", **SOLVE)
    output = {"X": np.asarray(X.value), "fun": np.array(problem.value)}
    return case(name, "sdp", [C, np.stack(a), np.array(b)], output, rtol=1e-5, atol=1e-5)


def lmi_case() -> Case:
    """Minimise the largest eigenvalue of A0 + x1 A1 + x2 A2: min t s.t. t I - A(x) PSD."""
    g = rng(31)
    mats: list[FloatArray] = []
    for _ in range(3):
        mi = g.normal(size=(3, 3))
        mats.append(mi + mi.T)
    x = cp.Variable(2)
    t = cp.Variable()
    expr = mats[0] + x[0] * mats[1] + x[1] * mats[2]
    constraints: list[cp.Constraint] = [cast(cp.Constraint, t * np.eye(3) - expr >> 0), cp.norm(x, "inf") <= 2]
    problem = cp.Problem(cp.Minimize(t), constraints)
    problem.solve(solver="CLARABEL", **SOLVE)
    xv = np.asarray(x.value, dtype=np.float64)
    output = {"x": np.array([xv[0], xv[1], float(np.asarray(t.value))]), "fun": np.array(problem.value)}
    return case("lmi/max-eigenvalue", "lmi", [np.stack(mats)], output, rtol=1e-5, atol=1e-5)


def cases() -> list[Case]:
    return [
        socp_case("socp/equality", 1, False),
        socp_case("socp/linear", 2, True),
        sdp_case("sdp/3x3", 3, 3, 2),
        sdp_case("sdp/4x4", 4, 4, 3),
        lmi_case(),
    ]
