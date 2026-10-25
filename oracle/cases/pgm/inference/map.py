"""MAP by LP relaxation against pgmpy's Mplp (generalised MPLP without triplet tightening) on pairwise Markov networks.

pgmpy's Mplp adds factor values as potentials (it maximises their sum), so the cases pass the log potentials theta
directly as factor values, kept positive by a constant shift, which leaves the maximiser unchanged."""

import numpy as np
import pgmpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.pgm import Potential, pgmpy_markov
from pgmpy.inference import Mplp

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}


def pairwise(
    g: np.random.Generator, cards: dict[str, int], edges: list[tuple[str, str]], scale: float
) -> list[Potential]:
    """Unary and pairwise log potentials theta + 3 (positive), rounded to 6 decimals."""
    pots: list[Potential] = []
    for v, k in cards.items():
        pots.append(([v], [round(float(x), 6) for x in 3 + g.normal(size=k) * 0.5]))
    for a, b in edges:
        size = cards[a] * cards[b]
        pots.append(([a, b], [round(float(x), 6) for x in 3 + g.normal(size=size) * scale]))
    return pots


def score(cards: dict[str, int], pots: list[Potential], x: dict[str, int]) -> float:
    """The sum of the potentials at an assignment."""
    total = 0.0
    for scope, vals in pots:
        flat = np.ravel_multi_index(tuple(x[v] for v in scope), tuple(cards[v] for v in scope))
        total += vals[int(flat)]
    return total


def brute_force(cards: dict[str, int], pots: list[Potential]) -> float:
    """The largest sum of potentials over every assignment."""
    names = list(cards)
    best = -np.inf
    for states in np.ndindex(*[cards[v] for v in names]):
        best = max(best, score(cards, pots, dict(zip(names, (int(s) for s in states), strict=True))))
    return float(best)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(53)
    names = [f"v{i}" for i in range(6)]
    tree_edges = [("v0", "v1"), ("v1", "v2"), ("v1", "v3"), ("v3", "v4"), ("v4", "v5")]
    loopy_edges = [*tree_edges, ("v0", "v2"), ("v2", "v5")]
    for label, edges, k, scale in [
        ("tree", tree_edges, 2, 1.0),
        ("tree-ternary", tree_edges, 3, 1.0),
        ("loopy", loopy_edges, 2, 0.8),
        ("loopy-ternary", loopy_edges, 3, 0.6),
    ]:
        cards = dict.fromkeys(names, k)
        pots = pairwise(g, cards, edges, scale)
        mn = pgmpy_markov(cards, pots)
        best = Mplp(mn).map_query(tighten_triplet=False, dual_threshold=1e-12, integrality_gap_threshold=1e-12)
        assignment = {v: int(best[v]) for v in names}
        # pgmpy's MPLP can stop at a suboptimal decoding (it does on the first tree); record whether its answer is
        # the brute-force optimum, so the test compares assignments only where pgmpy is right.
        optimal = bool(np.isclose(score(cards, pots, assignment), brute_force(cards, pots)))
        net = {
            "variables": [{"name": v, "states": [str(i) for i in range(k)]} for v in names],
            "potentials": [{"variables": s, "cardinality": [cards[v] for v in s], "theta": vals} for s, vals in pots],
        }
        out.append(case(label, "mplp", [], np.array(0.0), network=net, assignment=assignment, optimal=optimal))
    return out
