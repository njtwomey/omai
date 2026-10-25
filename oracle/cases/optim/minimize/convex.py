"""Subgradient, mirror descent and dual methods: converged solutions against cvxpy (Clarabel)."""

from importlib.metadata import version

import cvxpy as cp
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.optim_helpers import cvx_minimize

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy")}


def subgradient_cases() -> list[Case]:
    """min |A x - b|_1, a sharp polyhedral objective with a unique minimiser."""
    g = rng(51)
    a = g.normal(size=(15, 4))
    b = g.normal(size=15)
    x = cp.Variable(4)
    sol, value = cvx_minimize(cp.norm1(a @ x - b), x)
    return [case("l1Regression/0", "l1Regression", [a, b], {"x": sol, "fun": value}, rtol=1e-6, atol=1e-6)]


def mirror_cases() -> list[Case]:
    """Least squares over the simplex with an interior solution (exponentiated gradient converges linearly)."""
    g = rng(52)
    a = g.normal(size=(10, 5))
    interior = np.array([0.3, 0.1, 0.25, 0.15, 0.2])
    b = a @ interior + g.normal(size=10) * 0.05
    x = cp.Variable(5)
    sol, value = cvx_minimize(cp.sum_squares(a @ x - b) / 2, x, [x >= 0, cp.sum(x) == 1])
    return [
        case(
            "simplexLeastSquares/interior",
            "simplexLeastSquares",
            [a, b],
            {"x": sol, "fun": value},
            rtol=1e-6,
            atol=1e-6,
        )
    ]


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
    sol, value = cvx_minimize(cp.quad_form(x, p) / 2 + q @ x, x, [constraint])
    y = np.asarray(constraint.dual_value, dtype=np.float64)
    if np.linalg.norm(p @ sol + q + a.T @ y) > np.linalg.norm(p @ sol + q - a.T @ y):
        y = -y
    out = [case("equalityQp/0", "equalityQp", [p, q, a, b], {"x": sol, "y": y, "fun": value}, rtol=1e-6, atol=1e-6)]
    # A separable problem with one coupling constraint: sum_i (x_i - c_i)^2 w_i / 2 subject to sum x = 1.
    w = g.uniform(0.5, 2.0, size=4)
    c = g.normal(size=4)
    xs = cp.Variable(4)
    coupling = cp.sum(xs) == 1
    sol, value = cvx_minimize(cp.sum(cp.multiply(w, cp.square(xs - c))) / 2, xs, [coupling])
    out.append(case("separable/0", "separable", [w, c], {"x": sol, "fun": value}, rtol=1e-6, atol=1e-6))
    return out


def cases() -> list[Case]:
    return subgradient_cases() + mirror_cases() + dual_cases()
