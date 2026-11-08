"""Nash equilibria of bimatrix games against nashpy: support enumeration, vertex enumeration and Lemke-Howson from
every initial label, on textbook games (prisoner's dilemma, matching pennies, rock-paper-scissors, battle of the sexes,
stag hunt, Aumann's chicken, Shapley's game, hawk-dove), random non-degenerate games and a degenerate game. Equilibria
are not unique, so the cases store sets: the tests compare them sorted, up to tolerance."""

from collections.abc import Iterable
from typing import cast

import nashpy as nash
import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"nashpy": nash.__version__}

type Bimatrix = tuple[FloatArray, FloatArray]


def textbook() -> dict[str, Bimatrix]:
    """The classic games, as in omai-engine/games (normal-form/classic.ts)."""

    def zero_sum(a: list[list[float]]) -> Bimatrix:
        m = np.array(a, dtype=np.float64)
        return m, -m

    def pair(a: list[list[float]], b: list[list[float]]) -> Bimatrix:
        return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)

    return {
        "prisoners-dilemma": pair([[3, 0], [5, 1]], [[3, 5], [0, 1]]),
        "matching-pennies": zero_sum([[1, -1], [-1, 1]]),
        "rock-paper-scissors": zero_sum([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
        "battle-of-the-sexes": pair([[3, 0], [0, 2]], [[2, 0], [0, 3]]),
        "stag-hunt": pair([[4, 0], [3, 2]], [[4, 3], [0, 2]]),
        "chicken": pair([[6, 2], [7, 0]], [[6, 7], [2, 0]]),
        "shapley": pair([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        "hawk-dove": pair([[-1, 2], [0, 1]], [[-1, 0], [2, 1]]),
    }


def random_games() -> dict[str, Bimatrix]:
    """Generic games (non-degenerate with probability one) of several shapes, each the first seed with at least three
    equilibria (so that mixed equilibria and several Lemke-Howson endpoints occur)."""
    out: dict[str, Bimatrix] = {}
    for m, n in [(2, 2), (3, 3), (2, 4), (4, 3), (4, 4), (5, 5)]:
        for seed in range(500):
            g = rng(1000 * m + 10 * n + seed)
            a, b = g.uniform(-1, 1, size=(m, n)), g.uniform(-1, 1, size=(m, n))
            if len(list(nash.Game(a, b).support_enumeration())) >= 3:
                out[f"random-{m}x{n}"] = (a, b)
                break
    return out


def degenerate() -> dict[str, Bimatrix]:
    """von Stengel's (2007) degenerate example: a pure best response with two replies, and a continuum of equilibria."""
    return {
        "degenerate": (
            np.array([[3.0, 3.0], [2.0, 5.0], [0.0, 6.0]]),
            np.array([[3.0, 3.0], [2.0, 6.0], [3.0, 1.0]]),
        )
    }


def as_lists(equilibria: Iterable[tuple[object, object]]) -> list[list[list[float]]]:
    """Equilibria as nested lists, repeats removed (nashpy's halfspace intersection can repeat a vertex)."""
    out: list[list[list[float]]] = []
    for x, y in equilibria:
        xs = np.asarray(cast(npt.ArrayLike, x), dtype=np.float64)
        ys = np.asarray(cast(npt.ArrayLike, y), dtype=np.float64)
        pair = [[float(v) for v in xs], [float(v) for v in ys]]
        flat = np.array(pair[0] + pair[1])
        if not any(np.allclose(flat, np.array(o[0] + o[1]), atol=1e-9) for o in out):
            out.append(pair)
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    games = {**textbook(), **random_games()}
    for name, (a, b) in {**games, **degenerate()}.items():
        game = nash.Game(a, b)
        vertices = as_lists(list(game.vertex_enumeration()))
        out.append(
            case(f"{name}/vertex", "vertexEnumeration", [a, b], np.array(0.0), equilibria=vertices, rtol=0, atol=1e-8)
        )
        if name == "degenerate":
            continue
        # nashpy's support enumeration tests best responses by exact float equality, so it can drop mixed equilibria
        # of generic games; the case stores its output (a subset) and the complete set from vertex enumeration.
        support = as_lists(list(game.support_enumeration()))
        out.append(
            case(
                f"{name}/support",
                "supportEnumeration",
                [a, b],
                np.array(0.0),
                equilibria=support,
                all=vertices,
                rtol=0,
                atol=1e-9,
            )
        )
        labels: list[list[list[float]]] = []
        for label in range(sum(a.shape)):
            x, y = game.lemke_howson(initial_dropped_label=label)
            labels.append([[float(v) for v in x], [float(v) for v in y]])
        out.append(case(f"{name}/lemke-howson", "lemkeHowson", [a, b], np.array(0.0), labels=labels, atol=1e-9))
    return out
