"""CMA-ES against pycma (cma.fmin2), compared by the minima found: random streams differ between the libraries."""

from typing import Any, cast

import cma  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"cma": str(cma.__version__)}


def cmaes(name: str, problem: str, x0: list[float], sigma0: float) -> Case:
    options: dict[str, Any] = {"seed": 3, "verbose": -9, "tolfun": 1e-12, "tolx": 1e-12}
    xbest, es = cast(tuple[Any, Any], cma.fmin2(PROBLEMS[problem], x0, sigma0, options))
    out = np.concatenate([np.asarray(xbest, dtype=np.float64), [float(es.result.fbest)]])
    return case(name, "cmaes", [np.array(x0)], out, rtol=0, atol=1e-6, problem=problem, sigma0=sigma0, unique=True)


def cases() -> list[Case]:
    return [
        cmaes("cmaes/rosen4", "rosen", [0.0, 0.0, 0.0, 0.0], 0.5),
        cmaes("cmaes/beale", "beale", [1.0, 1.0], 0.5),
        cmaes("cmaes/quadratic", "quadratic", [3.0, -3.0, 3.0], 1.0),
    ]
