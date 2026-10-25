"""The sequence-form linear program against OpenSpiel's `sequence_form_lp.solve_zero_sum_game` (cvxpy) on Kuhn poker:
both players' values (the game value is -1/18) and the exploitability of the solution (zero up to the solver's
accuracy)."""

from importlib.metadata import version

import numpy as np
import pyspiel
from mdml_oracle.fixture import Case, case
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability, sequence_form_lp

REFERENCE: dict[str, str] = {"open_spiel": version("open_spiel")}


def cases() -> list[Case]:
    game = pyspiel.load_game("kuhn_poker")
    v0, v1, pi0, pi1 = sequence_form_lp.solve_zero_sum_game(game)
    merged = policy_lib.merge_tabular_policies([pi0, pi1], game)
    expl = float(exploitability.exploitability(game, merged))
    return [
        case(
            "kuhn_poker",
            "solveSequenceForm",
            [],
            np.array([float(v0), float(v1)]),
            exploitability=expl,
            rtol=1e-7,
            atol=1e-8,
        )
    ]
