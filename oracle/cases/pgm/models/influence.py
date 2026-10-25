"""Influence diagrams against pyAgrum (ShaferShenoyLIMIDInference with the no-forgetting assumption): maximum expected
utilities and optimal decisions of the oil wildcatter (Raiffa 1968) and of an umbrella problem with a forecast, and
values of information as differences of MEUs with and without an informational arc."""

from typing import Any

import numpy as np
import pyagrum as gum
import pyagrum.influence_diagram as idm
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"pyagrum": gum.__version__}

# Chance: (name, states, parents, rows); decision: (name, states, parents); utility: (name, parents, values).
type Chance = tuple[str, list[str], list[str], list[list[float]]]
type Decision = tuple[str, list[str], list[str]]
type Utility = tuple[str, list[str], list[float]]

OIL: tuple[list[Chance], list[Decision], list[Utility]] = (
    [
        ("oil", ["dry", "wet", "soaking"], [], [[0.5, 0.3, 0.2]]),
        (
            "seismic",
            ["none", "closed", "open", "diffuse"],
            ["oil", "test"],
            [[1, 0, 0, 0], [0, 0.1, 0.3, 0.6], [1, 0, 0, 0], [0, 0.3, 0.4, 0.3], [1, 0, 0, 0], [0, 0.5, 0.4, 0.1]],
        ),
    ],
    [("test", ["no", "yes"], []), ("drill", ["no", "yes"], ["seismic"])],
    [("cost", ["test"], [0, -10]), ("payoff", ["oil", "drill"], [0, -70, 0, 50, 0, 200])],
)

UMBRELLA: tuple[list[Chance], list[Decision], list[Utility]] = (
    [
        ("weather", ["dry", "rain"], [], [[0.7, 0.3]]),
        ("forecast", ["sunny", "cloudy", "rainy"], ["weather"], [[0.7, 0.2, 0.1], [0.15, 0.25, 0.6]]),
    ],
    [("umbrella", ["leave", "take"], ["forecast"])],
    [("u", ["weather", "umbrella"], [100, 70, 0, 80])],
)


def build(model: tuple[list[Chance], list[Decision], list[Utility]]) -> idm.InfluenceDiagram:
    chance, decisions, utilities = model
    states = {n: s for n, s, _, _ in chance} | {n: s for n, s, _ in decisions}
    d = idm.InfluenceDiagram()
    for n, s, _, _ in chance:
        d.addChanceNode(gum.LabelizedVariable(n, n, s))
    for n, s, _ in decisions:
        d.addDecisionNode(gum.LabelizedVariable(n, n, s))
    for n, _, _ in utilities:
        d.addUtilityNode(gum.LabelizedVariable(n, n, 1))
    for n, _, ps, _ in chance:
        for p in ps:
            d.addArc(p, n)
    for n, _, ps in decisions:
        for p in ps:
            d.addArc(p, n)
    for n, ps, _ in utilities:
        for p in ps:
            d.addArc(p, n)
    for n, _, ps, rows in chance:
        for r, row in enumerate(rows):
            d.cpt(n)[index(ps, states, r)] = row
    for n, ps, values in utilities:
        for r, v in enumerate(values):
            d.utility(n)[index(ps, states, r)] = [v]
    return d


def index(parents: list[str], states: dict[str, list[str]], r: int) -> dict[str, int]:
    """The parent assignment of row r (the first parent slowest)."""
    out: dict[str, int] = {}
    for p in reversed(parents):
        out[p] = r % len(states[p])
        r //= len(states[p])
    return out


def solve(model: tuple[list[Chance], list[Decision], list[Utility]]) -> tuple[float, dict[str, list[float]]]:
    d = build(model)
    ie = idm.ShaferShenoyLIMIDInference(d)
    ie.addNoForgettingAssumption([n for n, _, _ in model[1]])
    ie.makeInference()
    meu = float(ie.MEU()["mean"])
    decisions = {n: [float(v) for v in np.asarray(ie.optimalDecision(n).toarray()).ravel()] for n, _, _ in model[1]}
    return meu, decisions


def definition(model: tuple[list[Chance], list[Decision], list[Utility]]) -> dict[str, Any]:
    chance, decisions, utilities = model
    return {
        "chance": [
            {"name": n, "states": s, "parents": p, "cpt": [v for row in r for v in row]} for n, s, p, r in chance
        ],
        "decisions": [{"name": n, "states": s, "parents": p} for n, s, p in decisions],
        "utilities": [{"name": n, "parents": p, "values": v} for n, p, v in utilities],
    }


def cases() -> list[Case]:
    out: list[Case] = []
    for label, model in [("oil", OIL), ("umbrella", UMBRELLA)]:
        meu, _ = solve(model)
        out.append(case(f"{label}/meu", "meu", [], np.array(meu), rtol=1e-10, diagram=definition(model)))
    # Value of the forecast: the umbrella problem without the informational arc.
    chance, _, utilities = UMBRELLA
    blind = (chance, [("umbrella", ["leave", "take"], [])], utilities)
    with_info, _ = solve(UMBRELLA)
    without, _ = solve(blind)
    out.append(
        case(
            "umbrella/voi",
            "voi",
            [],
            np.array(with_info - without),
            rtol=1e-10,
            diagram=definition(blind),
            decision="umbrella",
            variables=["forecast"],
        )
    )
    return out
