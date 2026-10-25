"""The More-Thuente line search against scipy's line_search_wolfe1 (MINPACK-2 DCSRCH)."""

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray
from mdml_oracle.optim_problems import SMOOTH
from scipy.optimize import _linesearch as ls  # pyright: ignore[reportPrivateUsage]  (scipy's line searches)

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

# (problem, point, direction kind, previous value offset, c2)
SETUPS: list[tuple[str, FloatArray, str, float | None, float]] = [
    ("rosen", np.array([-1.2, 1.0]), "steepest", None, 0.9),
    ("rosen", np.array([-1.2, 1.0]), "steepest", 1.0, 0.9),
    ("rosen", np.array([0.5, 0.2]), "newton", None, 0.9),
    ("rosen", np.array([2.0, 2.0]), "steepest", None, 0.4),
    ("beale", np.array([1.0, 1.0]), "steepest", None, 0.9),
    ("beale", np.array([2.0, 0.2]), "steepest", 0.3, 0.1),
    ("himmelblau", np.array([0.0, 0.0]), "steepest", None, 0.9),
    ("spdQuadratic", np.array([1.0, 1.0, 1.0]), "steepest", None, 0.9),
    ("powell", np.array([3.0, -1.0, 0.0, 1.0]), "steepest", None, 0.9),
    ("wood", np.array([-3.0, -1.0, -3.0, -1.0]), "steepest", 5.0, 0.9),
    ("wood", np.array([-3.0, -1.0, -3.0, -1.0]), "scaled", None, 0.9),
]


def direction(name: str, x: FloatArray, kind: str) -> FloatArray:
    p = SMOOTH[name]
    g = p.grad(x)
    if kind == "newton":
        return -np.linalg.solve(p.hess(x), g)
    if kind == "scaled":
        return -g / np.linalg.norm(g)
    return -g


def nan_if_none(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, x, kind, offset, c2) in enumerate(SETUPS):
        prob = SMOOTH[name]
        p = direction(name, x, kind)
        f0 = prob.f(x)
        g0 = prob.grad(x)
        old = None if offset is None else f0 + offset
        params: dict[str, object] = {"problem": name, "c2": c2}
        if old is not None:
            params["previousValue"] = old

        a1, _fc1, _gc1, f1, _old1, _g1 = ls.line_search_wolfe1(prob.f, prob.grad, x, p, g0, f0, old, c2=c2)
        out.append(
            case(
                f"moreThuente/{i}",
                "moreThuente",
                [x, p],
                np.array([nan_if_none(a1), nan_if_none(f1)]),
                rtol=1e-10,
                atol=1e-12,
                **params,
            )
        )

    return out
