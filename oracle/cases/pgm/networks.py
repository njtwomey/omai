"""Bayes networks against pgmpy: posteriors by variable elimination, MAP queries, joint probabilities, d-separation and
Markov blankets, on the Asia and student networks."""

import numpy as np
import pgmpy
from pgmpy.inference import VariableElimination
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.pgm import (
    ASIA,
    STUDENT,
    NodeSpec,
    definition,
    evidence_arg,
    exact_posterior,
    hashables,
    order_arg,
    pgmpy_model,
    stable,
    ve_query,
)

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}


def hidden(spec: list[NodeSpec], variables: list[str], evidence: dict[str, str]) -> list[str]:
    """The variables to eliminate, in the network's order: an explicit order keeps pgmpy's results reproducible (its
    default heuristic breaks ties by set order, which varies between Python runs)."""
    return [n for n, _, _, _ in spec if n not in variables and n not in evidence]


QUERIES: dict[str, list[tuple[list[str], dict[str, str]]]] = {
    "asia": [
        (["lung"], {"xray": "yes"}),
        (["tub"], {"xray": "yes", "dysp": "yes", "asia": "yes"}),
        (["bronc", "lung"], {"dysp": "yes", "smoke": "yes"}),
        (["either"], {}),
    ],
    "student": [
        (["grade"], {}),
        (["intelligence"], {"grade": "C"}),
        (["intelligence"], {"grade": "C", "sat": "high"}),
        (["difficulty", "intelligence"], {"letter": "strong"}),
    ],
}

SEPARATIONS: list[tuple[str, str, list[str]]] = [
    ("asia", "smoke", []),
    ("tub", "lung", ["either"]),
    ("tub", "lung", []),
    ("xray", "dysp", ["either"]),
    ("asia", "dysp", ["either", "smoke"]),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for label, spec in [("asia", ASIA), ("student", STUDENT)]:
        model = pgmpy_model(spec)
        net = definition(spec)
        engine = VariableElimination(model)
        for i, (variables, evidence) in enumerate(QUERIES[label]):
            values = ve_query(model, variables, evidence, hidden(spec, variables, evidence))
            out.append(
                case(f"{label}/posterior/{i}", "posterior", [], values, network=net, query=variables, evidence=evidence)
            )
            found = engine.map_query(
                hashables(variables),
                evidence=evidence_arg(evidence),
                elimination_order=order_arg(hidden(spec, variables, evidence)),
                show_progress=False,
            )
            # In query order: pgmpy returns the assignment in set order, which varies between runs.
            best = {v: str(found[v]) for v in variables}
            out.append(
                case(
                    f"{label}/map/{i}",
                    "map",
                    [],
                    np.array(0.0),
                    network=net,
                    query=variables,
                    evidence=evidence,
                    assignment=best,
                )
            )
        names = [n for n, _, _, _ in spec]
        first = {n: s[0] for n, s, _, _ in spec}
        # The joint at the first state of every variable is the first entry of the joint in network order.
        p_first = stable(float(exact_posterior(model, names, {})[0]))
        out.append(case(f"{label}/joint/first", "joint", [], p_first, network=net, assignment=first))
        for node in names:
            blanket = sorted(str(v) for v in model.get_markov_blanket(node))
            out.append(
                case(f"{label}/blanket/{node}", "blanket", [], np.array(0.0), network=net, node=node, blanket=blanket)
            )
    model = pgmpy_model(ASIA)
    for x, y, z in SEPARATIONS:
        separated = not model.is_dconnected(x, y, observed=z)
        name = f"asia/dsep/{x}-{y}|{','.join(z)}"
        out.append(case(name, "dsep", [], np.array(float(separated)), network=definition(ASIA), x=x, y=y, z=z))
    return out
