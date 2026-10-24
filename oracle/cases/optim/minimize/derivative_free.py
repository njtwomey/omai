"""Nelder-Mead against scipy.optimize.minimize(method="Nelder-Mead") on standard problems.

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
    return out
