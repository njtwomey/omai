"""Shapley values and Banzhaf indices of textbook coalitional games, from their definitions: the Shapley value as the
average marginal contribution over every order of arrival (all n! orders for small games, the subset formula with
exact factorial weights for the UN Security Council), and the Banzhaf value as the average marginal contribution over
the coalitions of the others. Textbook games: glove games, the three-player majority game, weighted voting games
(including the Security Council, where a permanent member's Shapley-Shubik power is 0.1963), a bankruptcy game,
and random games."""

import itertools
import math
from collections.abc import Callable

import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"definition": "Shapley (1953), Banzhaf (1965); computed with math and itertools"}

type Worth = Callable[[list[int]], float]


def table(n: int, worth: Worth) -> FloatArray:
    """The worth of every coalition by bitmask."""
    out = np.zeros(2**n)
    for mask in range(1, 2**n):
        out[mask] = worth([i for i in range(n) if mask >> i & 1])
    return out


def glove(left: int) -> Worth:
    def worth(s: list[int]) -> float:
        lefts = sum(1 for i in s if i < left)
        return float(min(lefts, len(s) - lefts))

    return worth


def voting(quota: float, weights: list[float]) -> Worth:
    def worth(s: list[int]) -> float:
        return 1.0 if sum(weights[i] for i in s) >= quota else 0.0

    return worth


def bankruptcy(estate: float, claims: list[float]) -> Worth:
    def worth(s: list[int]) -> float:
        return max(0.0, estate - sum(c for i, c in enumerate(claims) if i not in s))

    return worth


def shapley_by_orders(n: int, v: FloatArray) -> list[float]:
    """The average marginal contribution over all n! orders."""
    phi = np.zeros(n)
    for order in itertools.permutations(range(n)):
        mask = 0
        for i in order:
            phi[i] += v[mask | 1 << i] - v[mask]
            mask |= 1 << i
    return [float(p) for p in phi / math.factorial(n)]


def shapley_by_subsets(n: int, v: FloatArray) -> list[float]:
    """The subset formula with exact weights |S|! (n - |S| - 1)! / n!."""
    phi = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                mask = sum(1 << j for j in subset)
                phi[i] += weight * float(v[mask | 1 << i] - v[mask])
    return phi


def banzhaf(n: int, v: FloatArray) -> list[float]:
    """The raw Banzhaf value."""
    beta = [0.0] * n
    for i in range(n):
        for mask in range(2**n):
            if not mask >> i & 1:
                beta[i] += float(v[mask | 1 << i] - v[mask])
    return [b / 2 ** (n - 1) for b in beta]


def cases() -> list[Case]:
    council = [7.0] * 5 + [1.0] * 10
    games: dict[str, tuple[int, Worth, dict[str, object]]] = {
        "glove-1-2": (3, glove(1), {"rule": "glove", "left": 1, "right": 2}),
        "glove-2-3": (5, glove(2), {"rule": "glove", "left": 2, "right": 3}),
        "majority-3": (3, voting(2, [1.0] * 3), {"rule": "voting", "quota": 2, "weights": [1.0] * 3}),
        "voting-51": (3, voting(51, [50.0, 49.0, 1.0]), {"rule": "voting", "quota": 51, "weights": [50.0, 49.0, 1.0]}),
        "voting-4": (
            4,
            voting(6, [4.0, 3.0, 2.0, 1.0]),
            {"rule": "voting", "quota": 6, "weights": [4.0, 3.0, 2.0, 1.0]},
        ),
        "security-council": (15, voting(39, council), {"rule": "voting", "quota": 39, "weights": council}),
        "bankruptcy": (
            3,
            bankruptcy(200, [100.0, 200.0, 300.0]),
            {"rule": "bankruptcy", "estate": 200, "claims": [100.0, 200.0, 300.0]},
        ),
    }
    out: list[Case] = []
    for name, (n, worth, params) in games.items():
        v = table(n, worth)
        phi = shapley_by_subsets(n, v) if n > 7 else shapley_by_orders(n, v)
        out.append(case(name, "shapleyValue", [], np.array(phi), n=n, banzhaf=banzhaf(n, v), game=params))
    for seed, n in enumerate([3, 4, 5, 6]):
        v = np.concatenate([[0.0], rng(1000 + seed).uniform(0, 1, size=2**n - 1)])
        out.append(
            case(
                f"random-{n}",
                "shapleyValue",
                [v],
                np.array(shapley_by_orders(n, v)),
                n=n,
                game={"rule": "table"},
                banzhaf=banzhaf(n, v),
            )
        )
    return out
