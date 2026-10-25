"""Best responses, NashConv and exploitability against OpenSpiel (`best_response.BestResponsePolicy`,
`exploitability.nash_conv` and `exploitability.exploitability`) on Kuhn and Leduc poker, for the uniform policy and
random policies (drawn here, stored by the engine's information-state keys)."""

from importlib.metadata import version
from typing import Protocol, cast

import numpy as np
import pyspiel
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import best_response, exploitability

REFERENCE: dict[str, str] = {"open_spiel": version("open_spiel")}


class SpielState(Protocol):
    """The parts of a pyspiel state used here (pyspiel ships no type information)."""

    def is_terminal(self) -> bool: ...
    def is_chance_node(self) -> bool: ...
    def chance_outcomes(self) -> list[tuple[int, float]]: ...
    def legal_actions(self) -> list[int]: ...
    def child(self, action: int) -> "SpielState": ...
    def history(self) -> list[int]: ...
    def current_player(self) -> int: ...
    def information_state_string(self) -> str: ...
    def apply_action(self, action: int) -> None: ...
    def returns(self) -> list[float]: ...


class NashConvReturn(Protocol):
    """OpenSpiel's `_NashConvReturn` named tuple."""

    nash_conv: float
    player_improvements: list[float]


class SpielGame(Protocol):
    """The parts of a pyspiel game used here."""

    def new_initial_state(self) -> SpielState: ...


LETTERS = {0: "f", 1: "c", 2: "r"}


def engine_key(name: str, game: SpielGame, state: SpielState) -> str:
    """The engine's information-state key of the player to move (OpenSpiel's own string for Kuhn poker)."""
    if name == "kuhn_poker":
        return state.information_state_string()
    player = state.current_player()
    replay = game.new_initial_state()
    cards: list[int] = []
    public = ""
    rounds: list[list[str]] = [[], []]
    for action in state.history():
        if replay.is_chance_node():
            if len(cards) < 2:
                cards.append(int(action))
            else:
                public = str(int(action))
        else:
            rounds[0 if public == "" else 1].append(LETTERS[int(action)])
        replay.apply_action(int(action))
    return f"{cards[player]}:{public}:{''.join(rounds[0])}/{''.join(rounds[1])}"


def keys(name: str, game: SpielGame) -> dict[str, tuple[str, int]]:
    """Engine key -> (OpenSpiel information state, number of legal actions), for every decision state."""
    out: dict[str, tuple[str, int]] = {}

    def walk(state: SpielState) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        out.setdefault(engine_key(name, game, state), (state.information_state_string(), len(state.legal_actions())))
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ("kuhn_poker", "leduc_poker"):
        game = pyspiel.load_game(name)
        table = keys(name, game)
        for label in ("uniform", "random-1", "random-2"):
            pol = policy_lib.TabularPolicy(game)
            engine: dict[str, list[float]] = {}
            g = rng(900 + len(out))
            for key, (spiel_key, k) in sorted(table.items()):
                probs = np.ones(k) / k if label == "uniform" else g.dirichlet(np.ones(k))
                engine[key] = [float(p) for p in probs]
                row = pol.policy_for_key(spiel_key)
                legal = np.nonzero(pol.legal_actions_mask[pol.state_lookup[spiel_key]])[0]
                row[:] = 0.0
                row[legal] = probs
            root = game.new_initial_state()
            br = [float(best_response.BestResponsePolicy(game, p, pol).value(root)) for p in range(2)]
            conv = cast(NashConvReturn, exploitability.nash_conv(game, pol, return_only_nash_conv=False))
            values = [float(v) for v in np.asarray(conv.player_improvements)]
            expl = float(exploitability.exploitability(game, pol))
            out.append(
                case(
                    f"{name}/{label}",
                    "bestResponse",
                    [],
                    np.array([br[0], br[1], float(conv.nash_conv), expl]),
                    game=name,
                    policy=engine if label != "uniform" else {},
                    improvements=values,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
    return out
