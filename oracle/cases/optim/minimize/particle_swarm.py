"""Particle swarm on standard problems: scipy has no particle swarm, so the references are the known global minima."""

import numpy as np
import scipy
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

# Global minimisers of the six-hump camel and Branin functions (each has several; one gives the value).
SIX_HUMP = np.array([0.08984201368301331, -0.7126564032704135])
BRANIN = np.array([np.pi, 2.275])


def known(name: str, problem: str, x0: list[float], bounds: list[tuple[float, float]], minimum: float) -> Case:
    """A case whose reference is the problem's known global minimum value."""
    return case(
        name,
        "particleSwarm",
        [np.array(x0)],
        np.array([minimum]),
        rtol=0,
        atol=1e-6,
        problem=problem,
        bounds=[list(b) for b in bounds],
    )


def cases() -> list[Case]:
    b2 = [(-5.0, 5.0), (-5.0, 5.0)]
    return [
        known("pso/ackley", "ackley", [2.0, 2.0], b2, 0.0),
        known("pso/himmelblau", "himmelblau", [0.0, 0.0], b2, 0.0),
        known(
            "pso/sixHumpCamel",
            "sixHumpCamel",
            [1.0, 1.0],
            [(-3.0, 3.0), (-2.0, 2.0)],
            PROBLEMS["sixHumpCamel"](SIX_HUMP),
        ),
        known("pso/branin", "branin", [0.0, 0.0], [(-5.0, 10.0), (0.0, 15.0)], PROBLEMS["branin"](BRANIN)),
    ]
