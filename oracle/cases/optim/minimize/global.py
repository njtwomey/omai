"""Basin hopping against scipy.optimize.basinhopping, compared by the minima found.

Random streams differ between the libraries, so cases compare the converged minimum (x where it is unique, the value
always). Simulated annealing has no scipy counterpart; its cases use the known global minima.
"""

import numpy as np
import scipy
import scipy.optimize as so
from mdml_oracle.fixture import Case, case
from mdml_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def hopping(name: str, problem: str, x0: list[float], niter: int) -> Case:
    res = so.basinhopping(PROBLEMS[problem], np.array(x0), niter=niter, rng=1)
    out = np.concatenate([np.asarray(res.x, dtype=np.float64), [float(res.fun)]])
    return case(name, "basinHopping", [np.array(x0)], out, rtol=0, atol=1e-6, problem=problem, niter=niter, unique=True)


def cases() -> list[Case]:
    return [
        hopping("bh/hop1", "hop1", [1.0], 100),
        hopping("bh/hop2", "hop2", [1.0, 1.0], 100),
        hopping("bh/himmelblau", "himmelblau", [0.0, 0.0], 30),
        case(
            "sa/himmelblau",
            "simulatedAnnealing",
            [np.array([0.0, 0.0])],
            np.array([0.0]),
            rtol=0,
            atol=1e-6,
            problem="himmelblau",
        ),
        case(
            "sa/sixHumpCamel",
            "simulatedAnnealing",
            [np.array([1.0, 1.0])],
            np.array([PROBLEMS["sixHumpCamel"](np.array([0.08984201368301331, -0.7126564032704135]))]),
            rtol=0,
            atol=1e-6,
            problem="sixHumpCamel",
            bounds=[[-3.0, 3.0], [-2.0, 2.0]],
        ),
    ]
