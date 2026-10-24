"""Belief propagation against pgmpy (BeliefPropagationWithMessagePassing on a tree-shaped factor graph, where it is
exact) and pyAgrum (LoopyBeliefPropagation, Pearl's message passing on loopy Bayes networks, whose fixed points are
those of sum-product on the networks' factor graphs). pyAgrum also stops when its epsilon stalls, which happens on
Asia after three iterations, so the loopy cases use networks where it converges."""

import numpy as np
import pgmpy
import pyagrum as gum
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagationWithMessagePassing
from pgmpy.models import FactorGraph
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.pgm import SPRINKLER, NodeSpec, Potential, definition, hashables, random_potentials, stable

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__, "pyagrum": gum.__version__}

TREE_CARDS: dict[str, int] = {"x0": 2, "x1": 3, "x2": 2, "x3": 2, "x4": 3, "x5": 2}
TREE_SCOPES: list[list[str]] = [["x0", "x1"], ["x1", "x2"], ["x1", "x3", "x4"], ["x4", "x5"], ["x0"], ["x5"]]
TREE_EVIDENCE: list[dict[str, int]] = [{}, {"x2": 1}, {"x3": 0, "x5": 1}]

# A loopy network with two undirected cycles and a three-parent node, CPT rows drawn from a seeded Dirichlet.
LOOPY_EDGES: list[tuple[str, list[str]]] = [
    ("a", []),
    ("b", ["a"]),
    ("c", ["a"]),
    ("d", ["b", "c"]),
    ("e", ["c"]),
    ("f", ["d", "e", "b"]),
]

LOOPY_QUERIES: dict[str, list[dict[str, str]]] = {
    "sprinkler": [{}, {"wet": "yes"}, {"wet": "yes", "rain": "no"}],
    "random": [{}, {"f": "s1"}, {"d": "s0", "e": "s1"}],
    "random3": [{}, {"f": "s2"}, {"a": "s1", "e": "s0"}],
}


def tree_graph(pots: list[Potential]) -> FactorGraph:
    fg = FactorGraph()
    fg.add_nodes_from(list(TREE_CARDS))
    factors = [DiscreteFactor(scope, [TREE_CARDS[v] for v in scope], np.array(vals)) for scope, vals in pots]
    fg.add_factors(*factors)
    fg.add_nodes_from(factors)
    fg.add_edges_from([(v, f) for f in factors for v in f.scope()])
    fg.check_model()
    return fg


def random_network(seed: int, k: int) -> list[NodeSpec]:
    """The loopy structure with k states per variable and CPT rows from a seeded Dirichlet(1, ..., 1)."""
    g = rng(seed)
    states = [f"s{i}" for i in range(k)]
    spec: list[NodeSpec] = []
    for name, parents in LOOPY_EDGES:
        rows: list[list[float]] = []
        for _ in range(k ** len(parents)):
            r = [round(float(p), 6) for p in g.dirichlet([1.0] * k)]
            rows.append([*r[:-1], round(1.0 - sum(r[:-1]), 6)])
        spec.append((name, states, parents, rows))
    return spec


def agrum(spec: list[NodeSpec]) -> gum.BayesNet:
    bn = gum.BayesNet("loopy")
    for name, states, _, _ in spec:
        bn.add(gum.LabelizedVariable(name, name, states))
    for name, _, parents, _ in spec:
        for p in parents:
            bn.addArc(p, name)
    sizes = {name: len(states) for name, states, _, _ in spec}
    for name, _, parents, rows in spec:
        for r, row in enumerate(rows):
            index: dict[str, int] = {}
            rest = r
            for p in reversed(parents):
                index[p] = rest % sizes[p]
                rest //= sizes[p]
            bn.cpt(name)[index] = row
    return bn


def loopy_posteriors(spec: list[NodeSpec], evidence: dict[str, str]) -> dict[str, list[float]]:
    ie = gum.LoopyBeliefPropagation(agrum(spec))
    ie.setEpsilon(1e-15)
    ie.setMinEpsilonRate(1e-15)
    ie.setMaxIter(100000)
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()
    names = [n for n, _, _, _ in spec if n not in evidence]
    return {n: [round(float(p), 12) for p in ie.posterior(n).tolist()] for n in names}


def cases() -> list[Case]:
    out: list[Case] = []
    pots = random_potentials(rng(2), TREE_CARDS, TREE_SCOPES, scale=0.9)
    fg = tree_graph(pots)
    net = {
        "variables": [{"name": v, "states": [str(i) for i in range(k)]} for v, k in TREE_CARDS.items()],
        "factors": [{"variables": s, "cardinality": [TREE_CARDS[v] for v in s], "values": vals} for s, vals in pots],
    }
    for i, evidence in enumerate(TREE_EVIDENCE):
        free = [v for v in TREE_CARDS if v not in evidence]
        result = BeliefPropagationWithMessagePassing(fg).query(hashables(free), evidence=dict(evidence))
        assert isinstance(result, dict)
        marginals: dict[str, list[float]] = {}
        for v in free:
            f = result[v]
            values = np.asarray(f.values, dtype=np.float64)
            marginals[v] = [float(x) for x in stable(values / values.sum())]
        out.append(case(f"tree/{i}", "tree", [], np.array(0.0), network=net, evidence=evidence, marginals=marginals))
    for label, spec in [("sprinkler", SPRINKLER), ("random", random_network(5, 2)), ("random3", random_network(8, 3))]:
        for i, evidence in enumerate(LOOPY_QUERIES[label]):
            marginals_l = loopy_posteriors(spec, evidence)
            out.append(
                case(
                    f"loopy/{label}/{i}",
                    "loopy",
                    [],
                    np.array(0.0),
                    rtol=1e-8,
                    atol=1e-10,
                    network=definition(spec),
                    evidence=evidence,
                    marginals=marginals_l,
                )
            )
    return out
