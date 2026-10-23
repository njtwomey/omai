"""SLSQP against scipy.optimize.minimize(method='SLSQP') on constrained test problems (unique solutions, compared
directly). The penalty, barrier and augmented Lagrangian methods share these problems (constrained.py)."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Fn = Callable[[FloatArray], float]
type Vec = Callable[[FloatArray], FloatArray]


def tutorial(x: FloatArray) -> float:
    return float((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2)


def tutorial_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] - 2 * x[1] + 2, -x[0] - 2 * x[1] + 6, -x[0] + 2 * x[1] + 2])


def hs071(x: FloatArray) -> float:
    return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])


def hs071_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] * x[1] * x[2] * x[3] - 25])


def hs071_eq(x: FloatArray) -> FloatArray:
    return np.array([np.sum(x * x) - 40])


def rosen(x: FloatArray) -> float:
    return float(so.rosen(x))


def rosen_ineq(x: FloatArray) -> FloatArray:
    return np.array([1 - x[0] - 2 * x[1], 1 - x[0] ** 2 - x[1], 1 - x[0] ** 2 + x[1]])


def rosen_eq(x: FloatArray) -> FloatArray:
    return np.array([1 - 2 * x[0] - x[1]])


def sphere(x: FloatArray) -> float:
    return float(np.sum(x * x))


def sphere_eq(x: FloatArray) -> FloatArray:
    return np.array([np.sum(x) - 1])


def sphere_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] - 0.5])


def disc(x: FloatArray) -> float:
    return float((x[0] - 2) ** 2 + (x[1] - 1) ** 2)


def disc_ineq(x: FloatArray) -> FloatArray:
    return np.array([1 - x[0] ** 2 - x[1] ** 2])


# name -> (objective, equalities, inequalities, bounds, start)
PROBLEMS: dict[str, tuple[Fn, Vec | None, Vec | None, list[tuple[float, float]] | None, list[float]]] = {
    "tutorial": (tutorial, None, tutorial_ineq, [(0.0, np.inf), (0.0, np.inf)], [2.0, 0.0]),
    "hs071": (hs071, hs071_eq, hs071_ineq, [(1.0, 5.0)] * 4, [1.0, 5.0, 5.0, 1.0]),
    "rosen": (rosen, rosen_eq, rosen_ineq, [(0.0, 1.0), (-0.5, 2.0)], [0.5, 0.0]),
    "sphere": (sphere, sphere_eq, sphere_ineq, None, [1.0, 1.0, 1.0]),
    "disc": (disc, None, disc_ineq, None, [0.0, 0.0]),
}


def solve(name: str) -> so.OptimizeResult:
    f, eq, ineq, bounds, x0 = PROBLEMS[name]
    constraints: list[so.NonlinearConstraint] = []
    if eq is not None:
        constraints.append(so.NonlinearConstraint(eq, 0.0, 0.0))
    if ineq is not None:
        constraints.append(so.NonlinearConstraint(ineq, 0.0, np.inf))
    return so.minimize(f, np.array(x0), method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": 1e-12})


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (_f, _eq, _ineq, bounds, x0) in PROBLEMS.items():
        r = solve(name)
        lower = [b[0] for b in bounds] if bounds is not None else []
        upper = [b[1] for b in bounds] if bounds is not None else []
        out.append(
            case(
                f"slsqp/{name}",
                "slsqp",
                [np.array(x0), np.array(lower), np.array(upper)],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
            )
        )
    return out
