"""Convex quadratic programs against cvxpy (Clarabel): solutions, optimal values and multipliers."""

from importlib.metadata import version

import cvxpy as cp
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng, spd

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy"), "clarabel": version("clarabel")}

type Program = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


def programs() -> dict[str, Program]:
    """name -> (P, q, G, h, A, b, lb, ub), with infinities for missing bounds."""
    inf = np.inf
    out: dict[str, Program] = {}
    out["nocedal-wright-16.4"] = (
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([-2.0, -5.0]),
        np.array([[1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0]]) * -1,
        np.array([2.0, 6.0, 2.0]),
        np.zeros((0, 2)),
        np.zeros(0),
        np.array([0.0, 0.0]),
        np.array([inf, inf]),
    )
    g = rng(21)
    n = 6
    x_feasible = g.uniform(-0.5, 0.5, size=n)
    G = g.normal(size=(8, n))
    A = g.normal(size=(2, n))
    out["random"] = (
        spd(g, n),
        g.normal(size=n) * 4,
        G,
        G @ x_feasible + g.uniform(0.0, 0.5, size=8),
        A,
        A @ x_feasible,
        np.full(n, -inf),
        np.full(n, inf),
    )
    out["box"] = (
        spd(g, n),
        g.normal(size=n) * 6,
        np.zeros((0, n)),
        np.zeros(0),
        np.zeros((0, n)),
        np.zeros(0),
        np.full(n, -1.0),
        np.full(n, 1.0),
    )
    out["equality"] = (
        spd(g, 4),
        g.normal(size=4),
        np.zeros((0, 4)),
        np.zeros(0),
        g.normal(size=(2, 4)),
        g.normal(size=2),
        np.full(4, -inf),
        np.full(4, inf),
    )
    out["semidefinite"] = (
        np.diag([1.0, 1.0, 0.0, 0.0]),
        np.array([-1.0, -3.0, 1.0, -1.0]),
        np.array([[1.0, 1.0, 0.0, 0.0]]),
        np.array([1.5]),
        np.zeros((0, 4)),
        np.zeros(0),
        np.zeros(4),
        np.full(4, 2.0),
    )
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (P, q, G, h, A, b, lb, ub) in programs().items():
        n = q.size
        x = cp.Variable(n)
        constraints: list[cp.Constraint] = []
        if G.size:
            constraints.append(G @ x <= h)
        if A.size:
            constraints.append(A @ x == b)
        lower = np.isfinite(lb)
        upper = np.isfinite(ub)
        if lower.any():
            constraints.append(x[lower] >= lb[lower])
        if upper.any():
            constraints.append(x[upper] <= ub[upper])
        problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x), constraints)
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        k = 0
        z = np.zeros(G.shape[0])
        y = np.zeros(A.shape[0])
        z_lower = np.zeros(n)
        z_upper = np.zeros(n)
        if G.size:
            z = np.asarray(constraints[k].dual_value)
            k += 1
        if A.size:
            y = np.asarray(constraints[k].dual_value)
            k += 1
        if lower.any():
            z_lower[lower] = np.asarray(constraints[k].dual_value)
            k += 1
        if upper.any():
            z_upper[upper] = np.asarray(constraints[k].dual_value)
        output = {
            "x": np.asarray(x.value),
            "fun": np.array(problem.value),
            "z": z,
            "y": y,
            "zLower": z_lower,
            "zUpper": z_upper,
        }
        out.append(case(name, "quadprog", [P, q, G, h, A, b, lb, ub], output, rtol=1e-7, atol=1e-7))
    return out
