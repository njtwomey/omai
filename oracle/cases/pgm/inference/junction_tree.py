"""Junction-tree inference against pgmpy's BeliefPropagation (clique-tree calibration): posteriors over one or several
variables (across cliques) and most probable explanations, on the Asia, student and sprinkler networks and a loopy
Markov network with random potentials."""

import numpy as np
import pgmpy
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.pgm import (
    NETWORKS,
    checked,
    definition,
    evidence_arg,
    exact_posterior,
    hashables,
    markov_definition,
    ordered,
    pgmpy_markov,
    pgmpy_model,
    random_potentials,
)

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

QUERIES: dict[str, list[tuple[list[str], dict[str, str]]]] = {
    "asia": [(["lung"], {"xray": "yes"}), (["tub", "bronc"], {"dysp": "yes"}), (["asia", "dysp"], {"xray": "no"})],
    "student": [(["intelligence"], {"grade": "C", "sat": "high"}), (["difficulty", "letter"], {})],
    "sprinkler": [(["rain"], {"wet": "yes"}), (["sprinkler", "rain"], {"wet": "yes"}), (["cloudy"], {"wet": "no"})],
}

CARDS: dict[str, int] = {"a": 2, "b": 3, "c": 2, "d": 2, "e": 2, "f": 3}
SCOPES: list[list[str]] = [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"], ["b", "d", "e"], ["e", "f"], ["f"], ["a"]]
MARKOV_QUERIES: list[tuple[list[str], dict[str, int]]] = [
    (["a"], {}),
    (["c", "f"], {}),
    (["b"], {"f": 2}),
    (["a", "e"], {"c": 1}),
]


def bp_query(
    model: DiscreteBayesianNetwork | DiscreteMarkovNetwork,
    variables: list[str],
    evidence: dict[str, str] | dict[str, int],
) -> np.ndarray:
    result = BeliefPropagation(model).query(hashables(variables), evidence=evidence_arg(evidence), show_progress=False)
    assert isinstance(result, DiscreteFactor)
    values = ordered(result, variables)
    return checked(values / values.sum(), exact_posterior(model, variables, evidence))


def cases() -> list[Case]:
    out: list[Case] = []
    for label, spec in NETWORKS.items():
        model = pgmpy_model(spec)
        net = definition(spec)
        for i, (variables, evidence) in enumerate(QUERIES[label]):
            out.append(
                case(
                    f"{label}/posterior/{i}",
                    "posterior",
                    [],
                    bp_query(model, variables, evidence),
                    network=net,
                    query=variables,
                    evidence=evidence,
                )
            )
        names = [n for n, _, _, _ in spec]
        best = BeliefPropagation(model).map_query(hashables(names), show_progress=False)
        assert isinstance(best, dict)
        assignment = {v: str(best[v]) for v in names}
        out.append(case(f"{label}/map", "map", [], np.array(0.0), network=net, assignment=assignment))
    pots = random_potentials(rng(11), CARDS, SCOPES, scale=0.7)
    mn = pgmpy_markov(CARDS, pots)
    net = markov_definition(CARDS, pots)
    for i, (variables, evidence) in enumerate(MARKOV_QUERIES):
        out.append(
            case(
                f"markov/posterior/{i}",
                "markovPosterior",
                [],
                bp_query(mn, variables, evidence),
                network=net,
                query=variables,
                evidence=evidence,
            )
        )
    return out
