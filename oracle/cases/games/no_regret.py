"""Regret matching against OpenSpiel's `regret_matching` (Hart and Mas-Colell, expected simultaneous updates, initial
regrets 1e-6 and uniform mixing gamma = 1e-6): the average strategies after a number of iterations, on two- and
three-player games."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from open_spiel.python.algorithms import regret_matching

REFERENCE: dict[str, str] = {"open_spiel": version("open_spiel")}


def cases() -> list[Case]:
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    games: dict[str, list[FloatArray]] = {
        "rock-paper-scissors": [rps, -rps],
        "battle-of-the-sexes": [np.array([[3.0, 0.0], [0.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])],
        "random-3x4": [rng(700).uniform(-1, 1, size=(3, 4)), rng(701).uniform(-1, 1, size=(3, 4))],
        "random-2x3x2": [rng(702 + p).uniform(-1, 1, size=(2, 3, 2)) for p in range(3)],
    }
    out: list[Case] = []
    for name, payoffs in games.items():
        for iterations in (1, 10, 200):
            average = regret_matching.regret_matching(payoffs, iterations=iterations, gamma=1e-6)
            out.append(
                case(
                    f"{name}/{iterations}",
                    "regretMatching",
                    payoffs,
                    np.array(0.0),
                    iterations=iterations,
                    average=[np.asarray(x, dtype=np.float64).tolist() for x in average],
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
    return out
