"""Total-variation proximal operators (1-D and 2-D, isotropic and anisotropic) against cvxpy."""

from importlib.metadata import version

import cvxpy as cp
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.optim_helpers import cvx_minimize

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy")}

TOL = 1e-6


def padded_differences(x: cp.Variable, m: int, n: int) -> tuple[cp.Expression, cp.Expression]:
    """Forward differences with zero beyond the last row and column (Beck and Teboulle's discretisation)."""
    rows = cp.vstack([x[1:, :] - x[:-1, :], np.zeros((1, n))])
    cols = cp.hstack([x[:, 1:] - x[:, :-1], np.zeros((m, 1))])
    return rows, cols


def tv_cases() -> list[Case]:
    g = rng(42)
    out: list[Case] = []
    for i, lam in enumerate([0.3, 1.0, 4.0]):
        y = np.repeat(g.normal(size=4) * 3, 5) + g.normal(size=20) * 0.4
        x = cp.Variable(20)
        sol, _ = cvx_minimize(cp.sum_squares(x - y) / 2 + lam * cp.norm1(cp.diff(x)), x)
        out.append(case(f"proxTv1d/{i}", "proxTv1d", [y], sol, rtol=TOL, atol=TOL, lam=lam))
    m, n = 6, 5
    image = np.kron(g.normal(size=(2, 1)) * 2, np.ones((3, 5))) + g.normal(size=(m, n)) * 0.3
    for isotropic in [True, False]:
        x = cp.Variable((m, n))
        rows, cols = padded_differences(x, m, n)
        if isotropic:
            tv = cp.sum(cp.norm(cp.vstack([cp.vec(rows, order="C"), cp.vec(cols, order="C")]), 2, axis=0))
        else:
            tv = cp.sum(cp.abs(rows)) + cp.sum(cp.abs(cols))
        sol, _ = cvx_minimize(cp.sum_squares(x - image) / 2 + 0.4 * tv, x)
        out.append(
            case(
                f"proxTv2d/{int(isotropic)}",
                "proxTv2d",
                [image],
                sol,
                rtol=1e-5,
                atol=1e-5,
                lam=0.4,
                isotropic=isotropic,
            )
        )
    return out


def cases() -> list[Case]:
    return tv_cases()
