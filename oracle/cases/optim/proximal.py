"""Proximal operators and projections against cvxpy (Clarabel at tight tolerances)."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import cast

import cvxpy as cp
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.optim_helpers import cvx_minimize

REFERENCE: dict[str, str] = {"cvxpy": version("cvxpy")}

TOL = 1e-6


def prox_case(name: str, point: FloatArray, term: object, x: cp.Variable, t: float, **params: object) -> Case:
    out, _ = cvx_minimize(cast(cp.Expression, term) * t + cp.sum_squares(x - point) / 2, x)
    return case(name, name.split("/")[0], [point], out, rtol=TOL, atol=TOL, t=t, **params)


def projection_case(
    name: str, point: FloatArray, x: cp.Variable, constraints: Sequence[object], **params: object
) -> Case:
    out, _ = cvx_minimize(cp.sum_squares(x - point), x, constraints)
    return case(name, name.split("/")[0], [point], out, rtol=TOL, atol=TOL, **params)


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
    sol, _ = cvx_minimize(cp.sum_squares(xa - v), xa, [a @ xa == b])
    out.append(case("projectAffine/0", "projectAffine", [v, a, b], sol, rtol=TOL, atol=TOL))
    hs = g.normal(size=7)
    sol, _ = cvx_minimize(cp.sum_squares(xa - v), xa, [hs @ xa <= -0.3])
    out.append(case("projectHalfspace/0", "projectHalfspace", [v, hs], sol, rtol=TOL, atol=TOL, b=-0.3))
    out.append(projection_case("projectNuclearBall/0", big, xm, [cp.normNuc(xm) <= 2.0], r=2.0))
    sym = g.normal(size=(4, 4))
    sym = (sym + sym.T) / 2
    xs = cp.Variable((4, 4), symmetric=True)
    out.append(projection_case("projectPsd/0", sym, xs, [xs >> 0]))
    return out


def cases() -> list[Case]:
    return operator_cases()
