"""Counterfactual regret minimisation against OpenSpiel's Python solvers (`CFRSolver`, `CFRPlusSolver`, `LCFRSolver`,
`DCFRSolver`): the average policy after a number of iterations and the exploitability (OpenSpiel's `exploitability`)
after every iteration, on Kuhn poker and (fewer iterations, a sample of the information sets) Leduc poker. Leduc's
information states are renamed to the engine's `card:public:round1/round2` keys."""

from importlib.metadata import version
from typing import Protocol

import numpy as np
import pyspiel
from omai_oracle.fixture import Case, case
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, discounted_cfr, exploitability

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


class SpielGame(Protocol):
    """The parts of a pyspiel game used here."""

    def new_initial_state(self) -> SpielState: ...


LETTERS = {0: "f", 1: "c", 2: "r"}


def engine_key(name: str, game: SpielGame, state: SpielState) -> str:
    """The engine's information-state key of the player to move."""
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


def policy_table(name: str, game: SpielGame, average: policy_lib.Policy) -> dict[str, list[float]]:
    """The average policy by engine key, over every decision state."""
    out: dict[str, list[float]] = {}

    def walk(state: SpielState) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        key = engine_key(name, game, state)
        if key not in out:
            probs = average.action_probabilities(state)
            out[key] = [float(probs.get(a, 0.0)) for a in state.legal_actions()]
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return out


def solver(variant: str, game: SpielGame) -> cfr._CFRSolver:  # pyright: ignore[reportPrivateUsage]
    if variant == "cfr":
        return cfr.CFRSolver(game)
    if variant == "cfr+":
        return cfr.CFRPlusSolver(game)
    if variant == "linear":
        return discounted_cfr.LCFRSolver(game)
    return discounted_cfr.DCFRSolver(game)


def cases() -> list[Case]:
    out: list[Case] = []
    runs = [("kuhn_poker", v, 50, 1) for v in ("cfr", "cfr+", "linear", "discounted")]
    runs += [("leduc_poker", v, 3, 20) for v in ("cfr", "cfr+")]
    for name, variant, iterations, every in runs:
        game = pyspiel.load_game(name)
        s = solver(variant, game)
        trace: list[float] = []
        for _ in range(iterations):
            s.evaluate_and_update_policy()
            trace.append(float(exploitability.exploitability(game, s.average_policy())))
        table = policy_table(name, game, s.average_policy())
        keys = sorted(table)[::every]
        out.append(
            case(
                f"{name}/{variant}",
                "cfr",
                [],
                np.array(trace),
                game=name,
                variant=variant,
                iterations=iterations,
                policy={k: table[k] for k in keys},
                rtol=1e-9,
                atol=1e-12,
            )
        )
    return out
