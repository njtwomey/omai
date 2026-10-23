"""Standard optimisation test problems, the same as packages/engine/src/testing/optim-problems.ts.

Each is written with the same operations in the same order as the TypeScript version (products rather than `**`, which
calls the platform's pow), so that evaluations agree to the last bit where the algorithms are deterministic.
"""

from collections.abc import Callable

import numpy as np
import scipy.optimize as so

from .helpers import FloatArray

type Problem = Callable[[FloatArray], float]


def rosen(x: FloatArray) -> float:
    return float(so.rosen(x))


def beale(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    xy = x * y
    a, b, c = (1.5 - x) + xy, (2.25 - x) + xy * y, (2.625 - x) + xy * y * y
    return a * a + b * b + c * c


def himmelblau(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    a, b = (x * x + y) - 11, (x + y * y) - 7
    return a * a + b * b


def ackley(x: FloatArray) -> float:
    n = len(x)
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
    b = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return float(a + b + 20 + np.e)


def rastrigin(x: FloatArray) -> float:
    return float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * len(x))


def branin(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    inner = y - b * x * x + c * x - 6
    return float(inner * inner + 10 * (1 - t) * np.cos(x) + 10)


def six_hump_camel(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    x2, y2 = x * x, y * y
    return (4 - 2.1 * x2 + x2 * x2 / 3) * x2 + x * y + (4 * y2 - 4) * y2


def quadratic(x: FloatArray) -> float:
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    return float(0.5 * x @ a @ x - b @ x)


def shifted(v: FloatArray) -> float:
    """(x - 1)^2 + (y - 2.5)^2, the objective of scipy's constrained-minimisation tutorial."""
    a, b = float(v[0]) - 1, float(v[1]) - 2.5
    return a * a + b * b


def hop1(x: FloatArray) -> float:
    """cos(14.5 x - 0.3) + (x + 0.2) x, scipy's basinhopping example in one dimension."""
    v = float(x[0])
    return float(np.cos(14.5 * v - 0.3) + (v + 0.2) * v)


def hop2(x: FloatArray) -> float:
    """cos(14.5 x0 - 0.3) + (x1 + 0.2) x1 + (x0 + 0.2) x0, scipy's basinhopping example in two dimensions."""
    a, b = float(x[0]), float(x[1])
    return float(np.cos(14.5 * a - 0.3) + (b + 0.2) * b + (a + 0.2) * a)


PROBLEMS: dict[str, Problem] = {
    "rosen": rosen,
    "beale": beale,
    "himmelblau": himmelblau,
    "ackley": ackley,
    "rastrigin": rastrigin,
    "branin": branin,
    "sixHumpCamel": six_hump_camel,
    "quadratic": quadratic,
    "shifted": shifted,
    "hop1": hop1,
    "hop2": hop2,
}
