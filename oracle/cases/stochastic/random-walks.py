"""Random walks: the n-step law of the simple walk (scipy.stats.binom), gambler's ruin probabilities and durations
(numpy.linalg.solve on the absorbing chain), self-avoiding walk counts (OEIS A001411 and A001412) and the stationary
law of walks on graphs (NetworkX degrees)."""

import networkx as nx
import numpy as np
import scipy
import scipy.stats as st
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "networkx": nx.__version__}

# Numbers of n-step self-avoiding walks on the square and cubic lattices (OEIS A001411, A001412).
SQUARE = [1, 4, 12, 36, 100, 284, 780, 2172, 5916, 16268, 44100]
CUBIC = [1, 6, 30, 150, 726, 3534, 16926]


def ruin(i: int, n: int, p: float) -> tuple[float, float]:
    """Win probability and expected duration by solving the absorbing chain's linear systems."""
    a = np.eye(n - 1)
    for k in range(1, n):
        if k + 1 < n:
            a[k - 1, k] -= p
        if k - 1 > 0:
            a[k - 1, k - 2] -= 1 - p
    b = np.zeros(n - 1)
    b[n - 2] = p
    win = np.linalg.solve(a, b)
    steps = np.linalg.solve(a, np.ones(n - 1))
    return float(win[i - 1]), float(steps[i - 1])


def cases() -> list[Case]:
    out: list[Case] = []
    for n, p in [(6, 0.5), (9, 0.3)]:
        k = np.arange(-n - 1, n + 2, dtype=np.float64)
        up = (n + k) / 2
        valid = (np.abs(k) <= n) & (np.mod(n + k, 2) == 0)
        pmf = np.where(valid, st.binom.pmf(np.where(valid, up, 0), n, p), 0.0)
        out.append(case(f"walk-pmf/{n}/{p}", "walk-pmf", [k], pmf, rtol=1e-12, atol=1e-15, steps=n, p=p))
    for i, n, p in [(3, 10, 0.5), (2, 7, 0.45), (5, 8, 0.6)]:
        win, steps = ruin(i, n, p)
        tag = f"{i}/{n}/{p}"
        out.append(case(f"ruin/win/{tag}", "ruin-win", [], np.array(win), rtol=1e-12, i=i, n=n, p=p))
        out.append(case(f"ruin/duration/{tag}", "ruin-duration", [], np.array(steps), rtol=1e-11, i=i, n=n, p=p))
    out.append(case("saw/square", "saw-count", [], np.array(SQUARE, dtype=np.float64), dim=2, n=len(SQUARE) - 1))
    out.append(case("saw/cubic", "saw-count", [], np.array(CUBIC, dtype=np.float64), dim=3, n=len(CUBIC) - 1))
    g = nx.Graph()
    edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 0.5), (2, 3, 3.0), (3, 4, 1.5)]
    g.add_weighted_edges_from(edges)
    deg = np.array([g.degree(v, weight="weight") for v in range(5)], dtype=np.float64)
    out.append(
        case(
            "graph-stationary",
            "graph-stationary",
            [],
            deg / deg.sum(),
            rtol=1e-12,
            edges=[[u, v, w] for u, v, w in edges],
        )
    )
    return out
