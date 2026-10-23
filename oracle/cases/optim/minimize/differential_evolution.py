"""Differential evolution against scipy.optimize.differential_evolution, compared by the minima found.

Random streams differ between the libraries, so cases compare the converged minimum (x where it is unique, the value
always) on problems whose global minimum both find. The TypeScript tests define the same problems
(packages/engine/src/testing/optim-problems.ts).
"""

from typing import Any, Literal

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Strategy = Literal["best1bin", "rand1exp", "best2bin", "randtobest1bin", "currenttobest1exp"]


def disc(x: FloatArray) -> FloatArray:
    """x^2 + y^2, bounded above by 1.5 in the constrained case."""
    return np.array([float(x[0]) ** 2 + float(x[1]) ** 2])


def de(
    name: str,
    problem: str,
    bounds: list[tuple[float, float]],
    unique: bool,
    strategy: Strategy = "best1bin",
    updating: Literal["immediate", "deferred"] = "immediate",
    constrained: bool = False,
) -> Case:
    constraints: Any = so.NonlinearConstraint(disc, -np.inf, 1.5) if constrained else ()
    res = so.differential_evolution(
        PROBLEMS[problem], bounds, strategy=strategy, updating=updating, rng=1, constraints=constraints, tol=1e-6
    )
    assert res.success, f"{name}: {res.message}"
    # Where the minimiser is not unique, only the minimum value is compared.
    out = np.concatenate([np.asarray(res.x, dtype=np.float64) if unique else [], [float(res.fun)]])
    return case(
        name,
        "differentialEvolution",
        [],
        out,
        rtol=0,
        atol=1e-5,
        problem=problem,
        bounds=[list(b) for b in bounds],
        unique=unique,
        strategy=strategy,
        updating=updating,
        constrained=constrained,
    )


def cases() -> list[Case]:
    b2 = [(-5.0, 5.0), (-5.0, 5.0)]
    return [
        de("de/ackley", "ackley", b2, True),
        de("de/rastrigin", "rastrigin", [(-5.12, 5.12), (-5.12, 5.12)], True),
        de("de/rosen3", "rosen", [(-2.0, 2.0)] * 3, True),
        de("de/branin", "branin", [(-5.0, 10.0), (0.0, 15.0)], False),
        de("de/sixHumpCamel", "sixHumpCamel", [(-3.0, 3.0), (-2.0, 2.0)], False),
        de("de/ackley-rand1exp-deferred", "ackley", b2, True, "rand1exp", "deferred"),
        de("de/rosen-best2bin", "rosen", [(-2.0, 2.0)] * 2, True, "best2bin"),
        de("de/himmelblau-randtobest1bin", "himmelblau", b2, False, "randtobest1bin"),
        de("de/beale-currenttobest1exp", "beale", [(-4.5, 4.5), (-4.5, 4.5)], True, "currenttobest1exp"),
        de("de/rosen-disc", "rosen", [(-2.0, 2.0), (-2.0, 2.0)], True, constrained=True),
    ]
