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


def model_factors(model: DiscreteBayesianNetwork | DiscreteMarkovNetwork) -> list[DiscreteFactor]:
    """The factors of a model in a fixed order: CPTs in node order, or potentials as added."""
    if isinstance(model, DiscreteBayesianNetwork):
        return [cast(TabularCPD, model.get_cpds(n)).to_factor() for n in model.nodes()]
    return [cast(DiscreteFactor, f) for f in model.get_factors()]


def state_index(model: DiscreteBayesianNetwork | DiscreteMarkovNetwork, variable: str, state: str | int) -> int:
    """The index of a state given by name (Bayes networks with state names) or index."""
    if isinstance(state, int):
        return state
    cpd = cast(TabularCPD, cast(DiscreteBayesianNetwork, model).get_cpds(variable))
    states: list[str] = [str(x) for x in cpd.state_names[variable]]
    return states.index(state)


def joint_table(model: DiscreteBayesianNetwork | DiscreteMarkovNetwork) -> tuple[list[str], FloatArray]:
    """The unnormalised joint as one array over the variables sorted by name, by one numpy einsum over the model's
    factor tables (deterministic, unlike pgmpy's engines, whose last bits vary with set order between Python runs)."""
    names = sorted(str(v) for v in model.nodes())
    letter = {v: chr(ord("a") + i) for i, v in enumerate(names)}
    factors = model_factors(model)
    tables = [np.asarray(f.values, dtype=np.float64) for f in factors]
    inputs = ",".join("".join(letter[str(v)] for v in f.variables) for f in factors)
    joint = np.einsum(f"{inputs}->{''.join(letter[v] for v in names)}", *tables)
    return names, np.asarray(joint, dtype=np.float64)


def exact_posterior(
    model: DiscreteBayesianNetwork | DiscreteMarkovNetwork,
    variables: Sequence[str],
    evidence: dict[str, str] | dict[str, int],
) -> FloatArray:
    """The posterior of some variables from the einsum joint of `joint_table`, reduced by the evidence and summed over
    the rest, axes in query order, normalised and flattened."""
    names, joint = joint_table(model)
    index = tuple(state_index(model, v, evidence[v]) if v in evidence else slice(None) for v in names)
    kept = [v for v in names if v not in evidence]
    table = joint[index]
    rest = tuple(i for i, v in enumerate(kept) if v not in variables)
    marginal = table.sum(axis=rest) if rest else table
    remaining = [v for v in kept if v in variables]
    values = np.transpose(marginal, [remaining.index(v) for v in variables]).ravel()
    return values / values.sum()


def checked(reference: FloatArray, exact: FloatArray) -> FloatArray:
    """Store the deterministic value after checking that the pgmpy engine agrees with it."""
    assert np.allclose(reference, exact, rtol=1e-10, atol=1e-14), (reference, exact)
    return stable(exact)


def ve_query(
    model: DiscreteBayesianNetwork | DiscreteMarkovNetwork,
    variables: list[str],
    evidence: dict[str, str] | dict[str, int],
    order: list[str] | None = None,
) -> FloatArray:
    """The posterior of some variables by pgmpy's variable elimination (elimination order defaulting to the remaining
    variables by name, as pgmpy's heuristic breaks ties by set order), checked against and stored as the fixed-order
    factor product of `exact_posterior`."""
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
    return checked(values / values.sum(), exact_posterior(model, variables, evidence))


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


def sample_rows(spec: list[NodeSpec], n: int, seed: int) -> list[dict[str, str]]:
    """Ancestral samples of a node list (inputs for the learners), drawn with a seeded numpy generator."""
    g = np.random.default_rng(seed)
    states = {name: s for name, s, _, _ in spec}
    rows: list[dict[str, str]] = []
    for _ in range(n):
        row: dict[str, str] = {}
        for name, s, parents, cpt in spec:
            j = 0
            for p in parents:
                j = j * len(states[p]) + states[p].index(row[p])
            row[name] = s[int(g.choice(len(s), p=np.array(cpt[j]) / np.sum(cpt[j])))]
        rows.append(row)
    return rows


def cpd_rows(cpd: TabularCPD, parents: Sequence[str]) -> list[float]:
    """A pgmpy CPD as whitebox-js's flat CPT over [...parents, node] (parents in the given order)."""
    values = np.asarray(cpd.get_values(), dtype=np.float64)  # [r, prod(parent cards)] in pgmpy's evidence order
    evidence = [str(v) for v in cpd.variables[1:]]
    r = values.shape[0]
    cards = [int(c) for c in cpd.cardinality[1:]]
    table = values.reshape([r, *cards])
    axes = [1 + evidence.index(p) for p in parents] + [0]
    return [float(v) for v in np.transpose(table, axes).ravel()]
