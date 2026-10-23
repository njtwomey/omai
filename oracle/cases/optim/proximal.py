"""Proximal operators, projections and total variation against cvxpy; proximal and splitting solvers against sklearn."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import cast

import cvxpy as cp
import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet, Lasso
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy"), "sklearn": sklearn.__version__}

TOL = 1e-6


def solve(objective: object, x: cp.Variable, constraints: Sequence[object] = ()) -> FloatArray:
    """Minimise with Clarabel at tight tolerances; return the value (cvxpy types its operators loosely)."""
    problem = cp.Problem(cp.Minimize(cast(cp.Expression, objective)), cast(list[cp.Constraint], list(constraints)))
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(x.value, dtype=np.float64)


def prox_case(name: str, point: FloatArray, term: object, x: cp.Variable, t: float, **params: object) -> Case:
    out = solve(cast(cp.Expression, term) * t + cp.sum_squares(x - point) / 2, x)
    return case(name, name.split("/")[0], [point], out, rtol=TOL, atol=TOL, t=t, **params)


def projection_case(
    name: str, point: FloatArray, x: cp.Variable, constraints: Sequence[object], **params: object
) -> Case:
    out = solve(cp.sum_squares(x - point), x, constraints)
    return case(name, name.split("/")[0], [point], out, rtol=TOL, atol=TOL, **params)


def padded_differences(x: cp.Variable, m: int, n: int) -> tuple[cp.Expression, cp.Expression]:
    """Forward differences with zero beyond the last row and column (Beck and Teboulle's discretisation)."""
    rows = cp.vstack([x[1:, :] - x[:-1, :], np.zeros((1, n))])
    cols = cp.hstack([x[:, 1:] - x[:, :-1], np.zeros((m, 1))])
    return rows, cols


def operator_cases() -> list[Case]:
    g = rng(41)
    out: list[Case] = []
    v = g.normal(size=7) * 2
    x = cp.Variable(7)
    out.append(prox_case("proxL1/0", v, cp.norm1(x), x, 0.7))
    out.append(prox_case("proxL2/0", v, cp.norm2(x), x, 1.3))
    out.append(prox_case("proxL2/1", v * 0.1, cp.norm2(x), x, 5.0))
    out.append(
        prox_case("proxElasticNet/0", v, 0.8 * cp.norm1(x) + 0.5 * 1.5 * cp.sum_squares(x), x, 0.9, l1=0.8, l2=1.5)
    )
    groups = [0, 0, 1, 1, 1, 2, 2]
    weights = [1.0, 2.0, 0.5]
    term = sum(w * cp.norm2(x[[i for i, gi in enumerate(groups) if gi == k]]) for k, w in enumerate(weights))
    out.append(prox_case("proxGroupLasso/0", v, term, x, 0.8, groups=groups, weights=weights))
    out.append(prox_case("proxLinf/0", v, cp.norm_inf(x), x, 1.1))
    huber = cp.sum(cp.huber(x, 1.2)) / 2  # pyright: ignore[reportArgumentType]  (the stub types M as int)
    out.append(prox_case("proxHuber/0", v, huber, x, 0.6, delta=1.2))
    big = g.normal(size=(4, 3)) * 2
    xm = cp.Variable((4, 3))
    out.append(prox_case("proxNuclear/0", big, cp.normNuc(xm), xm, 1.0))
    # Projections.
    out.append(projection_case("projectSimplex/0", v, x, [x >= 0, cp.sum(x) == 1]))
    out.append(projection_case("projectSimplex/1", v, x, [x >= 0, cp.sum(x) == 2.5], z=2.5))
    out.append(projection_case("projectL1Ball/0", v, x, [cp.norm1(x) <= 1.5], r=1.5))
    out.append(projection_case("projectL2Ball/0", v, x, [cp.norm2(x - 0.5) <= 1.0], r=1.0, center=0.5))
    out.append(projection_case("projectBox/0", v, x, [x >= -0.5, x <= 0.7], lo=-0.5, hi=0.7))
    a = g.normal(size=(3, 7))
    b = g.normal(size=3)
    xa = cp.Variable(7)
    sol = solve(cp.sum_squares(xa - v), xa, [a @ xa == b])
    out.append(case("projectAffine/0", "projectAffine", [v, a, b], sol, rtol=TOL, atol=TOL))
    hs = g.normal(size=7)
    sol = solve(cp.sum_squares(xa - v), xa, [hs @ xa <= -0.3])
    out.append(case("projectHalfspace/0", "projectHalfspace", [v, hs], sol, rtol=TOL, atol=TOL, b=-0.3))
    out.append(projection_case("projectNuclearBall/0", big, xm, [cp.normNuc(xm) <= 2.0], r=2.0))
    sym = g.normal(size=(4, 4))
    sym = (sym + sym.T) / 2
    xs = cp.Variable((4, 4), symmetric=True)
    out.append(projection_case("projectPsd/0", sym, xs, [xs >> 0]))
    return out


def tv_cases() -> list[Case]:
    g = rng(42)
    out: list[Case] = []
    for i, lam in enumerate([0.3, 1.0, 4.0]):
        y = np.repeat(g.normal(size=4) * 3, 5) + g.normal(size=20) * 0.4
        x = cp.Variable(20)
        sol = solve(cp.sum_squares(x - y) / 2 + lam * cp.norm1(cp.diff(x)), x)
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
        sol = solve(cp.sum_squares(x - image) / 2 + 0.4 * tv, x)
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


def lasso_cases() -> list[Case]:
    """min |A x - b|^2 / 2 + lam |x|_1: sklearn's Lasso with alpha = lam / m and no intercept."""
    g = rng(43)
    out: list[Case] = []
    m, n = 30, 8
    a = g.normal(size=(m, n))
    truth = np.array([2.0, 0.0, -1.5, 0.0, 0.0, 0.7, 0.0, 0.0])
    b = a @ truth + g.normal(size=m) * 0.1
    for i, lam in enumerate([0.5, 3.0]):
        model = Lasso(alpha=lam / m, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
        model.fit(a, b)
        coef = np.asarray(model.coef_, dtype=np.float64)
        out.append(case(f"lasso/{i}", "lasso", [a, b], coef, rtol=1e-6, atol=1e-7, lam=lam))
    net = ElasticNet(alpha=1.0 / m, l1_ratio=0.6, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
    net.fit(a, b)
    # sklearn: |b - A x|^2 / (2 m) + alpha l1_ratio |x|_1 + alpha (1 - l1_ratio) |x|^2 / 2; scaled by m.
    out.append(
        case(
            "elasticNet/0",
            "elasticNet",
            [a, b],
            np.asarray(net.coef_, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
            l1=0.6,
            l2=0.4,
        )
    )
    return out


def cases() -> list[Case]:
    return operator_cases() + tv_cases() + lasso_cases()
