"""Population methods against scipy.optimize.differential_evolution and pycma, compared by the minima they find.

Random streams differ between the libraries, so cases compare the converged minimum (x where it is unique, the value
always) on problems whose global minimum both find. Particle swarm has no reference implementation in scipy; its cases
use the known global minima. The TypeScript tests define the same problems (packages/engine/src/testing).
"""

from typing import Any, Literal, cast

import cma  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "cma": str(cma.__version__)}

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


def cmaes(name: str, problem: str, x0: list[float], sigma0: float) -> Case:
    options: dict[str, Any] = {"seed": 3, "verbose": -9, "tolfun": 1e-12, "tolx": 1e-12}
    xbest, es = cast(tuple[Any, Any], cma.fmin2(PROBLEMS[problem], x0, sigma0, options))
    out = np.concatenate([np.asarray(xbest, dtype=np.float64), [float(es.result.fbest)]])
    return case(name, "cmaes", [np.array(x0)], out, rtol=0, atol=1e-6, problem=problem, sigma0=sigma0, unique=True)


def known(name: str, fn: str, problem: str, x0: list[float], bounds: list[tuple[float, float]], minimum: float) -> Case:
    """A case whose reference is the problem's known global minimum value."""
    return case(
        name,
        fn,
        [np.array(x0)],
        np.array([minimum]),
        rtol=0,
        atol=1e-6,
        problem=problem,
        bounds=[list(b) for b in bounds],
    )


def cases() -> list[Case]:
    b2 = [(-5.0, 5.0), (-5.0, 5.0)]
    out = [
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
        cmaes("cmaes/rosen4", "rosen", [0.0, 0.0, 0.0, 0.0], 0.5),
        cmaes("cmaes/beale", "beale", [1.0, 1.0], 0.5),
        cmaes("cmaes/quadratic", "quadratic", [3.0, -3.0, 3.0], 1.0),
        known("pso/ackley", "particleSwarm", "ackley", [2.0, 2.0], b2, 0.0),
        known("pso/himmelblau", "particleSwarm", "himmelblau", [0.0, 0.0], b2, 0.0),
        known(
            "pso/sixHumpCamel",
            "particleSwarm",
            "sixHumpCamel",
            [1.0, 1.0],
            [(-3.0, 3.0), (-2.0, 2.0)],
            float(PROBLEMS["sixHumpCamel"](np.array([0.08984201368301331, -0.7126564032704135]))),
        ),
        known(
            "pso/branin",
            "particleSwarm",
            "branin",
            [0.0, 0.0],
            [(-5.0, 10.0), (0.0, 15.0)],
            float(PROBLEMS["branin"](np.array([np.pi, 2.275]))),
        ),
    ]
    return out
