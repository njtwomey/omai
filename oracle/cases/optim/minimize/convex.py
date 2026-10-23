"""Subgradient, mirror descent, Frank-Wolfe and dual methods: converged solutions against cvxpy (Clarabel)."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import cast

import cvxpy as cp
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy")}


def solve(objective: object, x: cp.Variable, constraints: Sequence[object] = ()) -> tuple[FloatArray, float]:
    """Minimise with Clarabel at tight tolerances; return the minimiser and the optimal value."""
    problem = cp.Problem(cp.Minimize(cast(cp.Expression, objective)), cast(list[cp.Constraint], list(constraints)))
    value = problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(x.value, dtype=np.float64), float(cast(float, value))


def subgradient_cases() -> list[Case]:
    """min |A x - b|_1, a sharp polyhedral objective with a unique minimiser."""
    g = rng(51)
    a = g.normal(size=(15, 4))
    b = g.normal(size=15)
    x = cp.Variable(4)
    sol, value = solve(cp.norm1(a @ x - b), x)
    return [case("l1Regression/0", "l1Regression", [a, b], {"x": sol, "fun": value}, rtol=1e-6, atol=1e-6)]


def simplex_cases() -> list[Case]:
    """Least squares over the simplex: one interior solution (mirror descent) and one on a face (Frank-Wolfe)."""
    g = rng(52)
    out: list[Case] = []
    a = g.normal(size=(10, 5))
    interior = np.array([0.3, 0.1, 0.25, 0.15, 0.2])
    b = a @ interior + g.normal(size=10) * 0.05
    x = cp.Variable(5)
    for name, rhs in [("interior", b), ("face", g.normal(size=10))]:
        sol, value = solve(cp.sum_squares(a @ x - rhs) / 2, x, [x >= 0, cp.sum(x) == 1])
        out.append(case(f"simplexLeastSquares/{name}", "simplexLeastSquares", [a, rhs], {"x": sol, "fun": value}))
    # Least squares over the L1 ball (a sparse solution on a face).
    rhs = g.normal(size=10) * 3
    sol, value = solve(cp.sum_squares(a @ x - rhs) / 2, x, [cp.norm1(x) <= 1.0])
    out.append(case("l1BallLeastSquares/0", "l1BallLeastSquares", [a, rhs], {"x": sol, "fun": value}, r=1.0))
    # Least squares over a box.
    sol, value = solve(cp.sum_squares(a @ x - rhs) / 2, x, [x >= -0.5, x <= 0.5])
    out.append(case("boxLeastSquares/0", "boxLeastSquares", [a, rhs], {"x": sol, "fun": value}, lo=-0.5, hi=0.5))
    # Matrix completion-like: |X - M|^2 / 2 over the nuclear ball.
    m = g.normal(size=(3, 4))
    xm = cp.Variable((3, 4))
    sol, value = solve(cp.sum_squares(xm - m) / 2, xm, [cp.normNuc(xm) <= 1.5])
    out.append(case("nuclearLeastSquares/0", "nuclearLeastSquares", [m], {"x": sol, "fun": value}, r=1.5))
    return [c.model_copy(update={"rtol": 1e-6, "atol": 1e-6}) for c in out]


def dual_cases() -> list[Case]:
    """min x^T P x / 2 + q . x subject to A x = b, with multipliers in the convention P x + q + A^T y = 0."""
    g = rng(53)
    n, m = 6, 2
    factor = g.normal(size=(n, n))
    p = factor @ factor.T + n * np.eye(n)
    q = g.normal(size=n)
    a = g.normal(size=(m, n))
    b = g.normal(size=m)
    x = cp.Variable(n)
    constraint = a @ x == b
    sol, value = solve(cp.quad_form(x, p) / 2 + q @ x, x, [constraint])
    y = np.asarray(constraint.dual_value, dtype=np.float64)
    if np.linalg.norm(p @ sol + q + a.T @ y) > np.linalg.norm(p @ sol + q - a.T @ y):
        y = -y
    out = [case("equalityQp/0", "equalityQp", [p, q, a, b], {"x": sol, "y": y, "fun": value}, rtol=1e-6, atol=1e-6)]
    # A separable problem with one coupling constraint: sum_i (x_i - c_i)^2 w_i / 2 subject to sum x = 1.
    w = g.uniform(0.5, 2.0, size=4)
    c = g.normal(size=4)
    xs = cp.Variable(4)
    coupling = cp.sum(xs) == 1
    sol, value = solve(cp.sum(cp.multiply(w, cp.square(xs - c))) / 2, xs, [coupling])
    out.append(case("separable/0", "separable", [w, c], {"x": sol, "fun": value}, rtol=1e-6, atol=1e-6))
    return out


def cases() -> list[Case]:
    return subgradient_cases() + simplex_cases() + dual_cases()
