"""Dynamic Bayes networks against pgmpy: filtered marginals by DBNInference.forward_inference (the interface algorithm)
and smoothed marginals by backward_inference, checked against and stored as the exact marginals of the unrolled
network (a fixed-order einsum), so the fixture does not vary with pgmpy's set order."""

from typing import Any

import numpy as np
import pgmpy
from omai_oracle.fixture import Case, case
from omai_oracle.pgm import NodeSpec, checked, definition, exact_posterior, pgmpy_model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

# Slice 0: (name, states, parents, rows). Transition: (name, states, parents, previous, rows), rows over
# [...previous, ...parents] assignments with the first varying slowest.
INITIAL: list[NodeSpec] = [
    ("z", ["a", "b"], [], [[0.6, 0.4]]),
    ("w", ["p", "q", "r"], [], [[0.3, 0.5, 0.2]]),
    ("x", ["u", "v"], ["z", "w"], [[0.9, 0.1], [0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]),
]
type TemplateSpec = tuple[str, list[str], list[str], list[str], list[list[float]]]
TRANSITION: list[TemplateSpec] = [
    ("z", ["a", "b"], [], ["z"], [[0.8, 0.2], [0.3, 0.7]]),
    (
        "w",
        ["p", "q", "r"],
        [],
        ["w", "z"],
        [[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]],
    ),
    ("x", ["u", "v"], ["z", "w"], [], [[0.9, 0.1], [0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]),
]
EVIDENCE: list[list[dict[str, str]]] = [
    [{"x": "u"}, {"x": "v"}, {"x": "v"}, {"x": "u"}],
    [{"x": "v", "w": "q"}, {}, {"x": "u"}],
]


def states_of(name: str) -> list[str]:
    return next(s for n, s, _, _ in INITIAL if n == name)


def pgmpy_dbn() -> DynamicBayesianNetwork:
    edges: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for n, _, ps, _ in INITIAL:
        edges += [((p, 0), (n, 0)) for p in ps]
    for n, _, _, prev, _ in TRANSITION:
        edges += [((p, 0), (n, 1)) for p in prev]
    dbn = DynamicBayesianNetwork(edges)
    cpds: list[TabularCPD] = []
    for n, s, ps, rows in INITIAL:
        cpds.append(
            TabularCPD(
                (n, 0),
                len(s),
                np.array(rows).T,
                evidence=[(p, 0) for p in ps] or None,
                evidence_card=[len(states_of(p)) for p in ps] or None,
            )
        )
    for n, s, ps, prev, rows in TRANSITION:
        if not prev:
            continue  # pgmpy copies slice-0 CPDs of nodes with no inter-slice parents
        parents: list[tuple[str, int]] = [(p, 0) for p in prev] + [(p, 1) for p in ps]
        cpds.append(
            TabularCPD(
                (n, 1),
                len(s),
                np.array(rows).T,
                evidence=parents,
                evidence_card=[len(states_of(p)) for p, _ in parents],
            )
        )
    dbn.add_cpds(*cpds)
    dbn.initialize_initial_state()
    return dbn


def unrolled(T: int) -> list[NodeSpec]:
    nodes: list[NodeSpec] = [(f"{n}_0", s, [f"{p}_0" for p in ps], rows) for n, s, ps, rows in INITIAL]
    for t in range(1, T):
        for n, s, ps, prev, rows in TRANSITION:
            nodes.append((f"{n}_{t}", s, [f"{p}_{t - 1}" for p in prev] + [f"{p}_{t}" for p in ps], rows))
    return nodes


def index_evidence(sequence: list[dict[str, str]]) -> dict[tuple[str, int], int]:
    return {(k, t): states_of(k).index(v) for t, ev in enumerate(sequence) for k, v in ev.items()}


def values(f: DiscreteFactor) -> np.ndarray:
    v = np.asarray(f.values, dtype=np.float64)
    return v / v.sum()


def cases() -> list[Case]:
    out: list[Case] = []
    dbn = pgmpy_dbn()
    inference = DBNInference(dbn)
    initial = definition(INITIAL)
    transition: list[dict[str, Any]] = [
        {"name": n, "states": s, "parents": ps, "previous": prev, "cpt": [v for r in rows for v in r]}
        for n, s, ps, prev, rows in TRANSITION
    ]
    for i, sequence in enumerate(EVIDENCE):
        T = len(sequence)
        filtered: list[dict[str, list[float]]] = []
        for t in range(T):
            prefix = sequence[: t + 1]
            model = pgmpy_model(unrolled(t + 1))
            flat = {f"{k}_{s}": v for s, ev in enumerate(prefix) for k, v in ev.items()}
            marg: dict[str, list[float]] = {}
            for name, _, _, _ in INITIAL:
                if name in sequence[t]:
                    continue
                exact = exact_posterior(model, [f"{name}_{t}"], flat)
                result = inference.forward_inference([(name, t)], index_evidence(prefix))
                assert isinstance(result, dict)
                marg[name] = [float(v) for v in checked(values(result[(name, t)]), exact)]
            filtered.append(marg)
        model = pgmpy_model(unrolled(T))
        flat = {f"{k}_{s}": v for s, ev in enumerate(sequence) for k, v in ev.items()}
        smoothed: dict[str, list[float]] = {}
        free = [n for n, _, _, _ in INITIAL if n not in sequence[0]]
        for name in free:
            exact = exact_posterior(model, [f"{name}_0"], flat)
            result = inference.backward_inference([(name, 0)], index_evidence(sequence))
            assert isinstance(result, dict)
            smoothed[name] = [float(v) for v in checked(values(result[(name, 0)]), exact)]
        out.append(
            case(
                f"filter/{i}",
                "filter",
                [],
                np.array(0.0),
                initial=initial,
                transition=transition,
                evidence=sequence,
                filtered=filtered,
                smoothed=smoothed,
            )
        )
    return out
