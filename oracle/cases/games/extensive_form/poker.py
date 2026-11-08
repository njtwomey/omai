"""Kuhn and Leduc poker against OpenSpiel's `kuhn_poker` and `leduc_poker`: the size of the game tree, the number of
information sets, and the returns of terminal histories (all of Kuhn's, every 50th of Leduc's, with checksums over all
of them). Histories list every action, chance deals included, as OpenSpiel's `state.history()`."""

from importlib.metadata import version
from typing import Protocol

import numpy as np
import pyspiel
from omai_oracle.fixture import Case, case

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


def walk(state: SpielState, nodes: list[int], infosets: set[str], leaves: list[tuple[list[int], float]]) -> None:
    """Count nodes, collect information states and terminal (history, player 0 return) pairs in depth-first order."""
    nodes[0] += 1
    if state.is_terminal():
        leaves.append(([int(a) for a in state.history()], float(state.returns()[0])))
        return
    if state.is_chance_node():
        for action, _ in state.chance_outcomes():
            walk(state.child(action), nodes, infosets, leaves)
        return
    infosets.add(state.information_state_string())
    for action in state.legal_actions():
        walk(state.child(action), nodes, infosets, leaves)


def cases() -> list[Case]:
    out: list[Case] = []
    for name, every in (("kuhn_poker", 1), ("leduc_poker", 50)):
        game = pyspiel.load_game(name)
        nodes = [0]
        infosets: set[str] = set()
        leaves: list[tuple[list[int], float]] = []
        walk(game.new_initial_state(), nodes, infosets, leaves)
        sample = [[h, r] for h, r in leaves[::every]]
        absolute = float(np.sum([abs(r) for _, r in leaves]))
        out.append(
            case(
                name,
                "pokerTree",
                [],
                np.array(0.0),
                nodes=nodes[0],
                infosets=len(infosets),
                terminals=len(leaves),
                absoluteReturns=absolute,
                every=every,
                sample=sample,
            )
        )
    return out
