"""COBYLA against scipy.optimize.minimize(method="COBYLA"), compared by the solution found.

The TypeScript tests define the same problems (packages/engine/src/testing/optim-problems.ts).
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from omai_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def constraint_x_le(v: FloatArray) -> float:
    """1 - x - y >= 0."""
    return 1.0 - float(v[0]) - float(v[1])


def constraint_disc(v: FloatArray) -> float:
    """1.5 - x^2 - y^2 >= 0."""
    return 1.5 - float(v[0]) ** 2 - float(v[1]) ** 2


def tutorial1(v: FloatArray) -> float:
    return float(v[0]) - 2 * float(v[1]) + 2


def tutorial2(v: FloatArray) -> float:
    return -float(v[0]) - 2 * float(v[1]) + 6


def tutorial3(v: FloatArray) -> float:
    return -float(v[0]) + 2 * float(v[1]) + 2


CONSTRAINTS: dict[str, Callable[[FloatArray], float]] = {
    "line": constraint_x_le,
    "disc": constraint_disc,
    "tutorial1": tutorial1,
    "tutorial2": tutorial2,
    "tutorial3": tutorial3,
}


def vector(c: Callable[[FloatArray], float]) -> Callable[[FloatArray], FloatArray]:
    """A scalar constraint as the 1-vector function NonlinearConstraint expects."""

    def g(x: FloatArray) -> FloatArray:
        return np.array([c(x)])

    return g


def cobyla(
    name: str,
    problem: str,
    x0: list[float],
    constraints: list[str],
    rhoend: float,
    maxfev: int = 1000,
    bounds: list[tuple[float, float]] | None = None,
) -> Case:
    cons = [so.NonlinearConstraint(vector(CONSTRAINTS[c]), 0.0, np.inf) for c in constraints]
    res = so.minimize(
        PROBLEMS[problem],
        np.array(x0),
        method="COBYLA",
        constraints=cons,
        bounds=bounds,
        options=cast(Any, {"rhobeg": 0.5, "tol": rhoend, "maxiter": maxfev}),
    )
    assert res.status == 0, f"{name}: scipy did not converge"
    extra: dict[str, object] = {} if bounds is None else {"bounds": [list(b) for b in bounds]}
    return case(
        name,
        "cobyla",
        [np.array(x0)],
        np.concatenate([np.asarray(res.x, dtype=np.float64), [float(res.fun)]]),
        rtol=0,
        atol=1e-5,
        problem=problem,
        constraints=constraints,
        rhoend=rhoend,
        maxfev=maxfev,
        **extra,
    )


def cases() -> list[Case]:
    out: list[Case] = []
    # COBYLA is compared by its solution: Powell's original and scipy's PRIMA port take different paths.
    out.append(cobyla("cobyla/0/rosen-disc", "rosen", [0.0, 0.0], ["disc"], 1e-8, maxfev=20000))
    out.append(cobyla("cobyla/1/quadratic-line", "quadratic", [0.0, 0.0, 0.0], [], 1e-8))
    out.append(cobyla("cobyla/2/himmelblau-line", "himmelblau", [0.0, 0.0], ["line"], 1e-8))
    out.append(cobyla("cobyla/3/beale-both", "beale", [0.0, 0.0], ["line", "disc"], 1e-8))
    out.append(
        cobyla(
            "cobyla/4/tutorial",
            "shifted",
            [2.0, 0.0],
            ["tutorial1", "tutorial2", "tutorial3"],
            1e-8,
            bounds=[(0.0, np.inf), (0.0, np.inf)],
        )
    )
    return out
