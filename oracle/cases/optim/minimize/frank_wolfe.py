"""Frank-Wolfe over the simplex, the L1 ball, a box and the nuclear ball: solutions against cvxpy (Clarabel)."""

from importlib.metadata import version

import cvxpy as cp
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.optim_helpers import cvx_minimize

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy")}


def simplex_cases() -> list[Case]:
    """Least squares over the simplex: one interior solution (mirror descent) and one on a face (Frank-Wolfe)."""
    g = rng(52)
    out: list[Case] = []
    a = g.normal(size=(10, 5))
    interior = np.array([0.3, 0.1, 0.25, 0.15, 0.2])
    b = a @ interior + g.normal(size=10) * 0.05
    x = cp.Variable(5)
    for name, rhs in [("interior", b), ("face", g.normal(size=10))]:
        sol, value = cvx_minimize(cp.sum_squares(a @ x - rhs) / 2, x, [x >= 0, cp.sum(x) == 1])
        out.append(case(f"simplexLeastSquares/{name}", "simplexLeastSquares", [a, rhs], {"x": sol, "fun": value}))
    # Least squares over the L1 ball (a sparse solution on a face).
    rhs = g.normal(size=10) * 3
    sol, value = cvx_minimize(cp.sum_squares(a @ x - rhs) / 2, x, [cp.norm1(x) <= 1.0])
    out.append(case("l1BallLeastSquares/0", "l1BallLeastSquares", [a, rhs], {"x": sol, "fun": value}, r=1.0))
    # Least squares over a box.
    sol, value = cvx_minimize(cp.sum_squares(a @ x - rhs) / 2, x, [x >= -0.5, x <= 0.5])
    out.append(case("boxLeastSquares/0", "boxLeastSquares", [a, rhs], {"x": sol, "fun": value}, lo=-0.5, hi=0.5))
    # Matrix completion-like: |X - M|^2 / 2 over the nuclear ball.
    m = g.normal(size=(3, 4))
    xm = cp.Variable((3, 4))
    sol, value = cvx_minimize(cp.sum_squares(xm - m) / 2, xm, [cp.normNuc(xm) <= 1.5])
    out.append(case("nuclearLeastSquares/0", "nuclearLeastSquares", [m], {"x": sol, "fun": value}, r=1.5))
    return [c.model_copy(update={"rtol": 1e-6, "atol": 1e-6}) for c in out]


def cases() -> list[Case]:
    return simplex_cases()
