"""Bayes-network graph queries against pgmpy: ancestors, active trails (Bayes ball), local independencies and the moral
graph, on the Asia, student and sprinkler networks."""

import numpy as np
import pgmpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import NETWORKS, definition, hashables, pgmpy_model
from pgmpy.independencies import IndependenceAssertion

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

TRAILS: dict[str, list[tuple[str, list[str]]]] = {
    "asia": [("asia", []), ("asia", ["either"]), ("tub", ["dysp"]), ("smoke", ["bronc", "lung"]), ("xray", ["tub"])],
    "student": [("difficulty", []), ("difficulty", ["grade"]), ("difficulty", ["letter"]), ("sat", ["intelligence"])],
    "sprinkler": [("sprinkler", []), ("sprinkler", ["cloudy"]), ("sprinkler", ["cloudy", "wet"])],
}

ANCESTORS: dict[str, list[list[str]]] = {
    "asia": [["dysp"], ["either", "bronc"], ["asia"]],
    "student": [["letter"], ["sat", "difficulty"]],
    "sprinkler": [["wet"]],
}


def assertion(a: IndependenceAssertion) -> dict[str, list[str]]:
    """An assertion as sorted variable lists."""
    return {
        "x": sorted(str(v) for v in a.event1),
        "y": sorted(str(v) for v in a.event2),
        "given": sorted(str(v) for v in a.event3),
    }


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    for label, spec in NETWORKS.items():
        model = pgmpy_model(spec)
        net = definition(spec)
        for start, observed in TRAILS[label]:
            reach = model.active_trail_nodes(hashables([start]), observed=hashables(observed))[start]
            nodes = sorted(str(v) for v in reach)
            name = f"{label}/trail/{start}|{','.join(observed)}"
            out.append(case(name, "trail", [], zero, network=net, start=start, observed=observed, nodes=nodes))
        for nodes in ANCESTORS[label]:
            anc = sorted(str(v) for v in model.get_ancestors(hashables(nodes)))
            out.append(
                case(f"{label}/ancestors/{','.join(nodes)}", "ancestors", [], zero, network=net, of=nodes, nodes=anc)
            )
        names = [n for n, _, _, _ in spec]
        local = [assertion(a) for a in model.local_independencies(hashables(names)).get_assertions()]
        local.sort(key=lambda a: a["x"])
        out.append(case(f"{label}/local", "local", [], zero, network=net, statements=local))
        moral = sorted(sorted((str(a), str(b))) for a, b in model.moralize().edges())
        out.append(case(f"{label}/moral", "moral", [], zero, network=net, edges=moral))
    return out
