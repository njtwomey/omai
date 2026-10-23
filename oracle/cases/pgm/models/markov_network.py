"""Markov networks against pgmpy: partition functions, posteriors by variable elimination and Markov blankets on a loopy
network with random potentials, and the Ising and Potts models on small grids."""

import itertools

import numpy as np
import pgmpy
from pgmpy.models import DiscreteMarkovNetwork
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.pgm import (
    Potential,
    checked,
    joint_table,
    markov_definition,
    pgmpy_markov,
    random_potentials,
    ve_query,
)

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

CARDS: dict[str, int] = {"a": 2, "b": 3, "c": 2, "d": 2, "e": 2}
SCOPES: list[list[str]] = [["a", "b"], ["b", "c", "d"], ["d", "a"], ["c", "e"], ["a"], ["e"]]

QUERIES: list[tuple[list[str], dict[str, int]]] = [
    (["a"], {}),
    (["b"], {}),
    (["b", "d"], {}),
    (["a"], {"e": 1}),
    (["c", "a"], {"b": 2}),
    (["d"], {"a": 0, "e": 0}),
]


def grid(rows: int, cols: int) -> tuple[list[str], list[tuple[str, str]]]:
    """Vertices s_r_c and the right and down edges of a grid, in whitebox-js's order."""
    name = [[f"s_{r}_{c}" for c in range(cols)] for r in range(rows)]
    vertices = [name[r][c] for r in range(rows) for c in range(cols)]
    edges: list[tuple[str, str]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((name[r][c], name[r][c + 1]))
            if r + 1 < rows:
                edges.append((name[r][c], name[r + 1][c]))
    return vertices, edges


def ising(rows: int, cols: int, j: float, h: float) -> tuple[dict[str, int], list[Potential]]:
    """Ising potentials exp(J s t) per edge and exp(h s) per vertex, spins (-1, +1) as states (0, 1)."""
    vertices, edges = grid(rows, cols)
    spins = [-1.0, 1.0]
    pots: list[Potential] = [([a, b], [float(np.exp(j * s * t)) for s in spins for t in spins]) for a, b in edges]
    pots += [([v], [float(np.exp(h * s)) for s in spins]) for v in vertices]
    return dict.fromkeys(vertices, 2), pots


def potts(rows: int, cols: int, q: int, j: float) -> tuple[dict[str, int], list[Potential]]:
    """Potts potentials exp(J [s = t]) per edge."""
    vertices, edges = grid(rows, cols)
    pots: list[Potential] = [
        ([a, b], [float(np.exp(j)) if s == t else 1.0 for s in range(q) for t in range(q)]) for a, b in edges
    ]
    return dict.fromkeys(vertices, q), pots


def log_partition(mn: DiscreteMarkovNetwork) -> FloatArray:
    """log Z from pgmpy's get_partition_function, checked against and stored as the einsum joint's total."""
    exact = np.log(np.sum(joint_table(mn)[1]))
    return checked(np.log(np.array(mn.get_partition_function())), np.array(exact))


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    pots = random_potentials(rng(3), CARDS, SCOPES, scale=0.8)
    mn = pgmpy_markov(CARDS, pots)
    net = markov_definition(CARDS, pots)
    out.append(case("loopy/logZ", "logZ", [], log_partition(mn), network=net))
    for i, (variables, evidence) in enumerate(QUERIES):
        values = ve_query(mn, variables, evidence)
        out.append(
            case(f"loopy/posterior/{i}", "posterior", [], values, network=net, query=variables, evidence=evidence)
        )
    for v in CARDS:
        blanket = sorted(str(x) for x in mn.markov_blanket(v))
        out.append(case(f"loopy/blanket/{v}", "blanket", [], zero, network=net, node=v, blanket=blanket))
    # The MPE of the network by enumeration of its normalised joint.
    values = ve_query(mn, list(CARDS), {})
    best = np.unravel_index(int(np.argmax(values)), tuple(CARDS.values()))
    out.append(
        case(
            "loopy/mpe",
            "mpe",
            [],
            zero,
            network=net,
            assignment={v: str(int(s)) for v, s in zip(CARDS, best, strict=True)},
        )
    )
    for label, (cards, model_pots), options in [
        ("ising-2x3", ising(2, 3, 0.4, 0.1), {"rows": 2, "cols": 3, "coupling": 0.4, "field": 0.1}),
        ("ising-3x3", ising(3, 3, -0.3, 0.0), {"rows": 3, "cols": 3, "coupling": -0.3}),
        ("potts-2x2", potts(2, 2, 3, 0.7), {"rows": 2, "cols": 2, "coupling": 0.7, "q": 3}),
    ]:
        model = pgmpy_markov(cards, model_pots)
        out.append(
            case(
                f"{label}/logZ",
                "latticeLogZ",
                [],
                log_partition(model),
                options=options,
                kind=label[:5],
            )
        )
        first = next(iter(cards))
        pair = list(itertools.islice(cards, 2))
        for variables in [[first], pair]:
            values = ve_query(model, variables, {})
            name = f"{label}/marginal/{','.join(variables)}"
            out.append(case(name, "latticeMarginal", [], values, options=options, kind=label[:5], query=variables))
    return out
