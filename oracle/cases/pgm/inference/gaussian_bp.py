"""Gaussian belief propagation against numpy: on tree-structured Gaussian Markov random fields the means and variances
are exact (numpy.linalg.solve and the diagonal of numpy.linalg.inv); on a loopy walk-summable field the means are."""

import networkx as nx
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "networkx": nx.__version__}


def tree_precision(g: np.random.Generator, n: int, seed: int) -> np.ndarray:
    """A diagonally dominant precision on the edges of a random labelled tree."""
    tree = nx.random_labeled_tree(n, seed=seed)
    q = np.zeros((n, n))
    for a, b in tree.edges():
        w = float(g.uniform(-1, 1))
        q[a, b] = q[b, a] = w
    q += np.diag(np.abs(q).sum(axis=1) + g.uniform(0.5, 1.5, size=n))
    return q


def grid_precision(g: np.random.Generator, k: int) -> np.ndarray:
    """A walk-summable precision on a k x k grid."""
    n = k * k
    q = np.zeros((n, n))
    for r in range(k):
        for c in range(k):
            i = r * k + c
            for j in [i + 1 if c + 1 < k else -1, i + k if r + 1 < k else -1]:
                if j >= 0:
                    w = float(g.uniform(-0.2, 0.2))
                    q[i, j] = q[j, i] = w
    return q + np.eye(n)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(23)
    for t, n in enumerate([6, 9, 12]):
        q = tree_precision(g, n, seed=t)
        mean = g.normal(size=n)
        observed = [n - 1] if t > 0 else []
        free = [i for i in range(n) if i not in observed]
        x = g.normal(size=len(observed))
        qa = q[np.ix_(free, free)]
        cond_mean = mean[free] - np.linalg.solve(qa, q[np.ix_(free, observed)] @ (x - mean[observed]))
        variances = np.diag(np.linalg.inv(qa))
        out.append(
            case(
                f"tree/{t}",
                "tree",
                [q, mean, x],
                np.concatenate([cond_mean, variances]),
                rtol=1e-10,
                atol=1e-12,
                observed=observed,
            )
        )
    q = grid_precision(g, 4)
    h = g.normal(size=16)
    out.append(case("grid", "grid", [q, h], np.linalg.solve(q, h), rtol=1e-8, atol=1e-10))
    return out
