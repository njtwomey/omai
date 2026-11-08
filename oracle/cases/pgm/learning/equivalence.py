"""Markov equivalence against pgmpy: immoralities (get_immoralities), CPDAGs (to_pdag, Chickering's compelled-edge
labelling), I-equivalence (is_iequivalent) and the structural Hamming distance (pgmpy.metrics.SHD), on seeded random
DAGs and on the student and Asia networks."""

import networkx as nx
import numpy as np
import pgmpy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.pgm import ASIA, STUDENT
from pgmpy.base import DAG
from pgmpy.metrics import SHD

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}


def random_dag(g: np.random.Generator, n: int, p: float) -> list[tuple[str, str]]:
    """Edges of a random DAG: a random order of v0..v{n-1}, each forward pair joined with probability p."""
    order = [f"v{i}" for i in g.permutation(n)]
    return [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if g.uniform() < p]


def reverse_covered(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reverse the first covered edge x -> y (parents of y = parents of x plus x), which keeps the DAG in its Markov
    equivalence class (Chickering 1995); unchanged when there is none."""
    parents: dict[str, set[str]] = {}
    for x, y in edges:
        parents.setdefault(y, set()).add(x)
    for k, (x, y) in enumerate(edges):
        if parents.get(y, set()) == parents.get(x, set()) | {x}:
            return [(y, x) if j == k else e for j, e in enumerate(edges)]
    return edges


def build(edges: list[tuple[str, str]], nodes: list[str]) -> DAG:
    d = DAG(edges)
    d.add_nodes_from(nodes)
    return d


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    g = rng(13)
    graphs: list[tuple[str, list[tuple[str, str]], list[str]]] = []
    for t in range(6):
        n = 6 + t % 3
        graphs.append((f"random/{t}", random_dag(g, n, 0.4), [f"v{i}" for i in range(n)]))
    for label, spec in [("student", STUDENT), ("asia", ASIA)]:
        graphs.append((label, [(p, n) for n, _, ps, _ in spec for p in ps], [n for n, _, _, _ in spec]))
    shd = SHD()
    for label, edges, nodes in graphs:
        d = build(edges, nodes)
        triples = sorted(
            [
                [min(str(a), str(b)), str(c), max(str(a), str(b))]
                for c, pairs in d.get_immoralities().items()
                for a, b in pairs
            ]
        )
        p = d.to_pdag()
        directed = sorted([[str(e[0]), str(e[1])] for e in p.directed_edges])
        undirected = sorted([sorted([str(e[0]), str(e[1])]) for e in p.undirected_edges])
        out.append(
            case(
                label,
                "graph",
                [],
                zero,
                edges=[list(e) for e in edges],
                nodes=nodes,
                immoralities=triples,
                directed=directed,
                undirected=undirected,
            )
        )
    # Equivalence and distance between pairs: a DAG against one of its reversals.
    for t in range(4):
        n = 6
        edges = random_dag(g, n, 0.45)
        nodes = [f"v{i}" for i in range(n)]
        flipped = [(b, a) if k % 2 == 0 else (a, b) for k, (a, b) in enumerate(edges)]
        if not nx.is_directed_acyclic_graph(nx.DiGraph(flipped)):
            flipped = [(b, a) if k == 0 else (a, b) for k, (a, b) in enumerate(edges)]
        a = build(edges, nodes)
        b = build(flipped, nodes) if t % 2 == 0 else build(reverse_covered(edges), nodes)
        flipped = [(str(x), str(y)) for x, y in b.edges()]
        out.append(
            case(
                f"pair/{t}",
                "pair",
                [],
                zero,
                a=[list(e) for e in edges],
                b=[list(e) for e in flipped],
                nodes=nodes,
                equivalent=bool(a.is_iequivalent(b)),
                shd=float(shd.evaluate(a, b)),
            )
        )
    return out
