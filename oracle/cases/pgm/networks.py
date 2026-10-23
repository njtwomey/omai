"""Bayes networks against pgmpy: posteriors by variable elimination, MAP queries, joint probabilities, d-separation and
Markov blankets, on the Asia and student networks."""

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pgmpy
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

# Each node: name, states, parents, and P(node | parents) as rows over the node's states, one row per joint parent
# assignment with the first parent varying slowest (whitebox-js's layout).
type NodeSpec = tuple[str, list[str], list[str], list[list[float]]]

ASIA: list[NodeSpec] = [
    ("asia", ["yes", "no"], [], [[0.01, 0.99]]),
    ("smoke", ["yes", "no"], [], [[0.5, 0.5]]),
    ("tub", ["yes", "no"], ["asia"], [[0.05, 0.95], [0.01, 0.99]]),
    ("lung", ["yes", "no"], ["smoke"], [[0.1, 0.9], [0.01, 0.99]]),
    ("bronc", ["yes", "no"], ["smoke"], [[0.6, 0.4], [0.3, 0.7]]),
    ("either", ["yes", "no"], ["lung", "tub"], [[1, 0], [1, 0], [1, 0], [0, 1]]),
    ("xray", ["yes", "no"], ["either"], [[0.98, 0.02], [0.05, 0.95]]),
    ("dysp", ["yes", "no"], ["bronc", "either"], [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]),
]

STUDENT: list[NodeSpec] = [
    ("difficulty", ["easy", "hard"], [], [[0.6, 0.4]]),
    ("intelligence", ["low", "high"], [], [[0.7, 0.3]]),
    (
        "grade",
        ["A", "B", "C"],
        ["intelligence", "difficulty"],
        [[0.3, 0.4, 0.3], [0.05, 0.25, 0.7], [0.9, 0.08, 0.02], [0.5, 0.3, 0.2]],
    ),
    ("sat", ["low", "high"], ["intelligence"], [[0.95, 0.05], [0.2, 0.8]]),
    ("letter", ["weak", "strong"], ["grade"], [[0.1, 0.9], [0.4, 0.6], [0.99, 0.01]]),
]


def pgmpy_model(spec: list[NodeSpec]) -> DiscreteBayesianNetwork:
    states = {name: s for name, s, _, _ in spec}
    model = DiscreteBayesianNetwork([(p, name) for name, _, parents, _ in spec for p in parents])
    model.add_nodes_from([name for name, _, _, _ in spec])
    for name, s, parents, rows in spec:
        model.add_cpds(
            TabularCPD(
                name,
                len(s),
                np.array(rows).T,  # pgmpy: rows are the node's states, columns the parent assignments
                evidence=parents or None,
                evidence_card=[len(states[p]) for p in parents] or None,
                state_names={name: s, **{p: states[p] for p in parents}},
            )
        )
    model.check_model()
    return model


def hidden(spec: list[NodeSpec], variables: list[str], evidence: dict[str, str]) -> list[str]:
    """The variables to eliminate, in the network's order: an explicit order keeps pgmpy's results bit-for-bit
    reproducible (its default heuristic breaks ties by set order, which varies between Python runs)."""
    return [n for n, _, _, _ in spec if n not in variables and n not in evidence]


# pgmpy accepts state names as evidence values but annotates query variables and evidence more narrowly
# (list[Hashable], dict[Hashable, int]); these helpers hold the casts in one place.
def query(
    engine: VariableElimination, spec: list[NodeSpec], variables: list[str], evidence: dict[str, str]
) -> DiscreteFactor:
    result = engine.query(
        cast(list[Hashable], variables),
        evidence=cast(dict[Hashable, int], evidence) if evidence else None,
        # pgmpy annotates elimination_order as str, though it accepts a list of variables.
        elimination_order=cast(Any, hidden(spec, variables, evidence)),
        show_progress=False,
    )
    assert isinstance(result, DiscreteFactor)
    return result


def map_query(
    engine: VariableElimination, spec: list[NodeSpec], variables: list[str], evidence: dict[str, str]
) -> dict[str, str]:
    best = engine.map_query(
        cast(list[Hashable], variables),
        evidence=cast(dict[Hashable, int], evidence) if evidence else None,
        # pgmpy annotates elimination_order as str, though it accepts a list of variables.
        elimination_order=cast(Any, hidden(spec, variables, evidence)),
        show_progress=False,
    )
    # In query order: pgmpy returns the assignment in set order, which varies between runs.
    return {v: str(best[v]) for v in variables}


def ordered(f: DiscreteFactor, variables: list[str]) -> FloatArray:
    """A factor's values with its axes in the requested variable order, flattened (pgmpy orders a multi-variable
    result by set iteration, which varies between runs)."""
    axes = [list(f.variables).index(v) for v in variables]
    return np.transpose(np.asarray(f.values, dtype=np.float64), axes).ravel()


def stable(x: npt.ArrayLike) -> FloatArray:
    """Round to 14 significant digits: pgmpy multiplies factors in set order, so the last bits of its results vary
    between Python runs. Fixtures are compared far more loosely than this, so rounding only makes them reproducible."""
    return np.array([float(f"{v:.14g}") for v in np.asarray(x, dtype=np.float64).ravel()]).reshape(np.shape(x))


def definition(spec: list[NodeSpec]) -> list[dict[str, Any]]:
    return [{"name": n, "states": s, "parents": p, "cpt": [v for row in rows for v in row]} for n, s, p, rows in spec]


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
            values = stable(ordered(query(engine, spec, variables, evidence), variables))
            out.append(
                case(f"{label}/posterior/{i}", "posterior", [], values, network=net, query=variables, evidence=evidence)
            )
            best = map_query(engine, spec, variables, evidence)
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
        p_first = stable(float(query(engine, spec, names, {}).get_value(**first)))
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
