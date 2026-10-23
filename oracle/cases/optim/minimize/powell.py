"""Powell's method against scipy.optimize.minimize(method="Powell") on standard problems.

The TypeScript tests define the same problems (packages/engine/src/testing/optim-problems.ts).
"""

from typing import Any, Literal, cast

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


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


def cases() -> list[Case]:
    out: list[Case] = []
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
    return out
