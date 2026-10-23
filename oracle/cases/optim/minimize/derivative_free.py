"""Derivative-free minimisation against scipy.optimize.minimize (Nelder-Mead, Powell, COBYLA) on standard problems.

The TypeScript tests define the same problems (packages/engine/src/testing/optim-problems.ts).
"""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Problem = Callable[[FloatArray], float]


def rosen(x: FloatArray) -> float:
    return float(so.rosen(x))


def beale(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    xy = x * y
    a, b, c = (1.5 - x) + xy, (2.25 - x) + xy * y, (2.625 - x) + xy * y * y
    # The same operations as the TypeScript problem (products, not pow), so evaluations agree to the last bit.
    return a * a + b * b + c * c


def himmelblau(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    a, b = (x * x + y) - 11, (x + y * y) - 7
    return a * a + b * b


def quadratic(x: FloatArray) -> float:
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    return float(0.5 * x @ a @ x - b @ x)


def shifted(v: FloatArray) -> float:
    """(x - 1)^2 + (y - 2.5)^2, the objective of scipy's constrained-minimisation tutorial."""
    a, b = float(v[0]) - 1, float(v[1]) - 2.5
    return a * a + b * b


PROBLEMS: dict[str, Problem] = {
    "rosen": rosen,
    "beale": beale,
    "himmelblau": himmelblau,
    "quadratic": quadratic,
    "shifted": shifted,
}


def result(res: Any) -> dict[str, object]:  # noqa: ANN401  (scipy's OptimizeResult is untyped)
    return {"x": np.asarray(res.x, dtype=np.float64), "fun": float(res.fun), "nfev": int(res.nfev)}


def run(
    name: str,
    fn: str,
    problem: str,
    x0: list[float],
    method: Literal["Nelder-Mead", "Powell"],
    rtol: float,
    atol: float,
    options: dict[str, Any],
    bounds: list[tuple[float, float]] | None = None,
    params: dict[str, object] | None = None,
) -> Case:
    res = so.minimize(PROBLEMS[problem], np.array(x0), method=method, bounds=bounds, options=cast(Any, options))
    out = result(res)
    extra: dict[str, object] = dict(params or {})
    if bounds is not None:
        extra["bounds"] = [list(b) for b in bounds]
    return case(
        name,
        fn,
        [np.array(x0)],
        np.concatenate([out["x"], [out["fun"]]]),  # type: ignore[list-item]
        rtol=rtol,
        atol=atol,
        problem=problem,
        nfev=out["nfev"],
        **extra,
    )


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
    # Nelder-Mead follows scipy's moves exactly, so iterates (and evaluation counts) agree to rounding.
    nm: list[tuple[str, list[float], dict[str, Any], list[tuple[float, float]] | None]] = [
        ("rosen", [-1.2, 1.0], {}, None),
        ("rosen", [-1.2, 1.0], {"xatol": 1e-8, "fatol": 1e-10, "maxfev": 2000}, None),
        ("rosen", [0.0, 0.0, 0.0, 0.0], {"maxiter": 3000, "maxfev": 3000}, None),
        ("beale", [1.0, 1.0], {"xatol": 1e-8, "fatol": 1e-10}, None),
        ("himmelblau", [0.0, 0.0], {}, None),
        ("quadratic", [0.0, 0.0, 0.0], {"adaptive": True, "xatol": 1e-8, "fatol": 1e-10}, None),
        ("rosen", [-1.2, 1.0], {"xatol": 1e-8, "fatol": 1e-10}, [(-2.0, 0.8), (-1.0, 0.7)]),
        ("rosen", [-1.2, 1.0], {"maxfev": 40}, None),
    ]
    for i, (problem, x0, options, bounds) in enumerate(nm):
        params = {k: v for k, v in options.items()}
        out.append(
            run(
                f"nelderMead/{i}/{problem}",
                "nelderMead",
                problem,
                x0,
                "Nelder-Mead",
                1e-9,
                1e-11,
                options,
                bounds,
                params,
            )
        )
    # Powell: the same bracket, Brent and bounded line minimisations as scipy.
    pw: list[tuple[str, list[float], dict[str, Any], list[tuple[float, float]] | None]] = [
        ("rosen", [-1.2, 1.0], {}, None),
        ("rosen", [-1.2, 1.0], {"xtol": 1e-8, "ftol": 1e-12}, None),
        ("beale", [1.0, 1.0], {"xtol": 1e-8, "ftol": 1e-12}, None),
        ("quadratic", [0.0, 0.0, 0.0], {}, None),
        ("himmelblau", [0.0, 0.0], {"xtol": 1e-8, "ftol": 1e-12}, [(-5.0, 5.0), (-5.0, 1.5)]),
        ("rosen", [0.5, 0.5], {"xtol": 1e-8, "ftol": 1e-12}, [(0.0, np.inf), (-np.inf, 0.8)]),
    ]
    for i, (problem, x0, options, bounds) in enumerate(pw):
        params = {k: v for k, v in options.items()}
        out.append(run(f"powell/{i}/{problem}", "powell", problem, x0, "Powell", 1e-7, 1e-9, options, bounds, params))
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
