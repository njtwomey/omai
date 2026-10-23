"""Shared pieces for the pgm case modules: example networks in whitebox-js's layout, their pgmpy models, and the
helpers that make pgmpy's results reproducible (explicit elimination orders, query-ordered axes, rounding)."""

from collections.abc import Hashable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork

from .helpers import FloatArray

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

# The sprinkler network (cloudy -> sprinkler, rain -> wet grass): a loop in the moral graph.
SPRINKLER: list[NodeSpec] = [
    ("cloudy", ["no", "yes"], [], [[0.5, 0.5]]),
    ("sprinkler", ["off", "on"], ["cloudy"], [[0.5, 0.5], [0.9, 0.1]]),
    ("rain", ["no", "yes"], ["cloudy"], [[0.8, 0.2], [0.2, 0.8]]),
    ("wet", ["no", "yes"], ["sprinkler", "rain"], [[1.0, 0.0], [0.1, 0.9], [0.1, 0.9], [0.01, 0.99]]),
]

NETWORKS: dict[str, list[NodeSpec]] = {"asia": ASIA, "student": STUDENT, "sprinkler": SPRINKLER}


def pgmpy_model(spec: list[NodeSpec]) -> DiscreteBayesianNetwork:
    """The pgmpy network of a node list, with state names."""
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


def definition(spec: list[NodeSpec]) -> list[dict[str, Any]]:
    """A node list as whitebox-js `BayesNode`s."""
    return [{"name": n, "states": s, "parents": p, "cpt": [v for row in rows for v in row]} for n, s, p, rows in spec]


def stable(x: npt.ArrayLike) -> FloatArray:
    """Round to 14 significant digits: pgmpy multiplies factors in set order, so the last bits of its results vary
    between Python runs. Fixtures are compared far more loosely than this, so rounding only makes them reproducible."""
    return np.array([float(f"{v:.14g}") for v in np.asarray(x, dtype=np.float64).ravel()]).reshape(np.shape(x))


def ordered(f: DiscreteFactor, variables: Sequence[str]) -> FloatArray:
    """A factor's values with its axes in the requested variable order, flattened (pgmpy orders a multi-variable
    result by set iteration, which varies between runs)."""
    axes = [list(f.variables).index(v) for v in variables]
    return np.transpose(np.asarray(f.values, dtype=np.float64), axes).ravel()


def hashables(variables: Sequence[str]) -> list[Hashable]:
    """pgmpy annotates variable lists as list[Hashable]."""
    return list(variables)


def evidence_arg(evidence: dict[str, str] | dict[str, int]) -> dict[Hashable, int] | None:
    """pgmpy accepts state names as evidence values but annotates evidence as dict[Hashable, int]."""
    return cast(dict[Hashable, int], evidence) if evidence else None


def order_arg(order: list[str]) -> str:
    """pgmpy annotates elimination_order as str, though it accepts a list of variables."""
    return cast(str, order)


def ve_query(
    model: DiscreteBayesianNetwork | DiscreteMarkovNetwork,
    variables: list[str],
    evidence: dict[str, str] | dict[str, int],
    order: list[str] | None = None,
) -> FloatArray:
    """The posterior of some variables by pgmpy's variable elimination, axes in query order, normalised, flattened and
    rounded. The
    elimination order defaults to the remaining variables sorted by name (pgmpy's own heuristic breaks ties by set
    order)."""
    hidden = (
        order
        if order is not None
        else [str(v) for v in sorted(model.nodes()) if v not in variables and v not in evidence]
    )
    result = VariableElimination(model).query(
        hashables(variables), evidence=evidence_arg(evidence), elimination_order=order_arg(hidden), show_progress=False
    )
    assert isinstance(result, DiscreteFactor)
    values = ordered(result, variables)
    return stable(values / values.sum())  # pgmpy leaves Markov-network posteriors unnormalised


# A Markov network: variables with cardinalities, and potentials as (scope, flat row-major values).
type Potential = tuple[list[str], list[float]]


def pgmpy_markov(cards: dict[str, int], potentials: list[Potential]) -> DiscreteMarkovNetwork:
    """The pgmpy Markov network of some potentials (edges from their scopes)."""
    edges = {
        tuple(sorted((a, b))) for scope, _ in potentials for i, a in enumerate(scope) for b in scope[i + 1 :] if a != b
    }
    mn = DiscreteMarkovNetwork(sorted(edges))
    mn.add_nodes_from(sorted(cards))
    for scope, values in potentials:
        mn.add_factors(DiscreteFactor(scope, [cards[v] for v in scope], np.array(values)))
    mn.check_model()
    return mn


def markov_definition(cards: dict[str, int], potentials: list[Potential]) -> dict[str, Any]:
    """A Markov network as whitebox-js data: variables with states "0".."k-1" and potentials with their values."""
    return {
        "variables": [{"name": v, "states": [str(i) for i in range(k)]} for v, k in cards.items()],
        "potentials": [
            {"variables": scope, "cardinality": [cards[v] for v in scope], "values": vals} for scope, vals in potentials
        ],
    }


def random_potentials(
    g: np.random.Generator, cards: dict[str, int], scopes: list[list[str]], scale: float = 1.0
) -> list[Potential]:
    """Positive potentials exp(scale * N(0, 1)) over some scopes, rounded so the inputs are short and exact."""
    out: list[Potential] = []
    for scope in scopes:
        size = int(np.prod([cards[v] for v in scope]))
        out.append((scope, [round(float(v), 6) for v in np.exp(scale * g.normal(size=size))]))
    return out
