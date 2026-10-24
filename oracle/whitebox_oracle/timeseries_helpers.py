"""Shared series for the matrix-profile cases: noisy oscillations with a planted repeated pattern and an anomaly."""

import numpy as np
import numpy.typing as npt

from .helpers import FloatArray, rng


def series(n: int, seed: int) -> FloatArray:
    """A noisy two-tone oscillation: no ties, no constant windows."""
    g = rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    return np.sin(t) + 0.5 * np.sin(2.3 * t + 0.4) + 0.3 * g.normal(size=n)


def planted(n: int, seed: int, m: int) -> tuple[FloatArray, list[int], int]:
    """A random walk with one pattern planted twice (a motif pair) and a spike (a discord); returns the series, the
    motif starts and the discord start."""
    g = rng(seed)
    t = np.cumsum(g.normal(size=n)) * 0.3
    pattern = np.sin(np.linspace(0, 2 * np.pi, m)) * 3.0
    starts = [n // 5, (3 * n) // 5]
    for s in starts:
        t[s : s + m] = t[s] + pattern + 0.01 * g.normal(size=m)
    discord = (4 * n) // 5
    t[discord : discord + m // 2] += 4.0 * g.normal(size=m // 2)
    return t, starts, discord


def as_float(a: npt.ArrayLike) -> FloatArray:
    """A flat float64 copy (STUMPY returns object arrays for profiles and matches)."""
    return np.asarray(a, dtype=float).ravel()


def ideal_arc_curve(width: int, bidirectional: bool) -> FloatArray:
    """The exact expectation of the random-neighbour arc counts (the engine's idealArcCurve)."""
    x = np.arange(width, dtype=float)
    w = float(width)
    if bidirectional:
        return 2 * (x + 1) * (w - 1 - x) / w
    out = np.zeros(width)
    for i in range(width - 1):
        out[i] = (w - 1 - i) * np.sum(1.0 / (w - 1 - np.arange(i + 1)))
    return out
