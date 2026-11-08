"""Fictitious play against nashpy's `fictitious_play`: the play counts after every round, from random positive initial
counts (nashpy breaks ties at random; generic initial counts make every best response unique, so the trajectories are
deterministic), on rock-paper-scissors, Shapley's game (where fictitious play cycles), the battle of the sexes and
random games."""

import nashpy as nash
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"nashpy": nash.__version__}


def cases() -> list[Case]:
    games: dict[str, tuple[FloatArray, FloatArray]] = {
        "rock-paper-scissors": (
            np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]),
            -np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]),
        ),
        "shapley": (
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        ),
        "battle-of-the-sexes": (np.array([[3.0, 0.0], [0.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])),
        "random-3x4": (rng(500).uniform(-1, 1, size=(3, 4)), rng(501).uniform(-1, 1, size=(3, 4))),
        "random-4x4": (rng(502).uniform(-1, 1, size=(4, 4)), rng(503).uniform(-1, 1, size=(4, 4))),
    }
    out: list[Case] = []
    for seed, (name, (a, b)) in enumerate(games.items()):
        g = rng(600 + seed)
        initial = [g.uniform(0.1, 1.0, size=a.shape[0]), g.uniform(0.1, 1.0, size=a.shape[1])]
        counts = list(nash.Game(a, b).fictitious_play(iterations=60, play_counts=[c.copy() for c in initial]))
        plays = [[int(np.argmax(counts[t + 1][p] - counts[t][p])) for p in range(2)] for t in range(len(counts) - 1)]
        final = [counts[-1][0].tolist(), counts[-1][1].tolist()]
        out.append(
            case(
                name,
                "fictitiousPlay",
                [a, b],
                np.array(0.0),
                initial=[c.tolist() for c in initial],
                plays=plays,
                counts=final,
            )
        )
    return out
