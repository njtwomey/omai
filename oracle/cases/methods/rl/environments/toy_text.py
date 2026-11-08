"""Toy-text environments against gymnasium: Taxi's transition tables and Blackjack's rules with planted cards."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__}


def table(env: gym.Env[Any, Any], states: list[int]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The outcomes of `env.P` for some states as rows `[state, action, probability, next, reward, terminated]`."""
    p_table = cast(Any, env).P
    rows: list[list[float]] = []
    for s in states:
        for a in sorted(p_table[s]):
            for p, s2, r, t in p_table[s][a]:
                rows.append([float(s), float(a), float(p), float(s2), float(r), float(t)])
    return np.array(rows, dtype=np.float64)


class PlantedCards:
    """Stands in for `np_random` so that Blackjack draws a given sequence of cards."""

    def __init__(self, cards: list[int]) -> None:
        self.cards = list(cards)

    def choice(self, deck: list[int]) -> int:
        return self.cards.pop(0)


def blackjack(
    player: list[int], dealer: list[int], action: int, cards: list[int], natural: bool, sab: bool
) -> list[float]:
    """The observation, reward and termination after one action from given hands, with planted draws."""
    env = cast(Any, gym.make("Blackjack-v1", natural=natural, sab=sab).unwrapped)
    env.reset(seed=0)
    env.player = list(player)
    env.dealer = list(dealer)
    env.np_random = PlantedCards(cards)
    obs, reward, terminated, _, _ = env.step(action)
    return [float(obs[0]), float(obs[1]), float(obs[2]), float(reward), float(terminated)]


def cases() -> list[Case]:
    out: list[Case] = []
    configs: list[tuple[str, dict[str, Any]]] = [
        ("dry", {}),
        ("rainy", {"is_rainy": True}),
        ("rainy-0.6", {"is_rainy": True, "rainy_probability": 0.6}),
    ]
    # Every action of a sample of states (the corners, the locations and a random spread) keeps the fixture small.
    picked = sorted({int(s) for s in rng(72).choice(500, size=60, replace=False)} | {0, 99, 400, 499, 328})
    for name, kwargs in configs:
        env = gym.make("Taxi-v4", **kwargs).unwrapped
        out.append(case(f"taxi/{name}/table", "taxiTable", [], table(env, picked), **kwargs))
        initial = np.asarray(cast(Any, env).initial_state_distrib, dtype=np.float64)
        out.append(case(f"taxi/{name}/initial", "taxiInitial", [], initial, **kwargs))
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    g = rng(71)
    hands: list[tuple[list[int], list[int]]] = [
        ([1, 10], [10, 7]),
        ([1, 10], [1, 10]),
        ([1, 10], [9, 9]),
        ([10, 6], [1, 5]),
    ]
    for _ in range(26):
        hands.append(([int(g.choice(deck)), int(g.choice(deck))], [int(g.choice(deck)), int(g.choice(deck))]))
    for i, (player, dealer) in enumerate(hands):
        cards = [int(g.choice(deck)) for _ in range(12)]
        for action in (0, 1):
            for natural, sab in [(False, False), (True, False), (False, True)]:
                got = blackjack(player, dealer, action, cards, natural, sab)
                out.append(
                    case(
                        f"blackjack/{i}/{action}/{int(natural)}{int(sab)}",
                        "blackjack",
                        [],
                        np.array(got),
                        player=player,
                        dealer=dealer,
                        action=action,
                        cards=cards,
                        natural=natural,
                        sab=sab,
                    )
                )
    return out
