"""Standard optimisation test problems, the same as packages/engine/src/testing/optim-problems.ts.

`PROBLEMS` holds objectives by name; `SMOOTH` holds the smooth problems of the gradient-based methods with analytic
gradients, Hessians and standard starting points.

Each is written with the same operations in the same order as the TypeScript version (products rather than `**`, which
calls the platform's pow), so that evaluations agree to the last bit where the algorithms are deterministic.
"""

from collections.abc import Callable
from dataclasses import dataclass

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


QUADRATIC_A: FloatArray = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.5], [0.5, -0.5, 2.0]])
QUADRATIC_B: FloatArray = np.array([1.0, -2.0, 0.5])


@dataclass(frozen=True)
class Smooth:
    """A test problem: objective, gradient, Hessian and the standard starting point."""

    f: Callable[[FloatArray], float]
    grad: Callable[[FloatArray], FloatArray]
    hess: Callable[[FloatArray], FloatArray]
    x0: FloatArray


def rosen_der(x: FloatArray) -> FloatArray:
    return np.asarray(so.rosen_der(x), dtype=np.float64)


def rosen_hess(x: FloatArray) -> FloatArray:
    return np.asarray(so.rosen_hess(x), dtype=np.float64)


def beale_der(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y**2, 2.625 - x + x * y**3
    return np.array(
        [
            2 * a * (y - 1) + 2 * b * (y**2 - 1) + 2 * c * (y**3 - 1),
            2 * a * x + 2 * b * 2 * x * y + 2 * c * 3 * x * y**2,
        ]
    )


def beale_hess(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y**2, 2.625 - x + x * y**3
    ax, ay = y - 1, x
    bx, by = y**2 - 1, 2 * x * y
    cx, cy = y**3 - 1, 3 * x * y**2
    hxx = 2 * (ax * ax + bx * bx + cx * cx)
    hxy = 2 * (ax * ay + a + bx * by + b * 2 * y + cx * cy + c * 3 * y**2)
    hyy = 2 * (ay * ay + by * by + b * 2 * x + cy * cy + c * 6 * x * y)
    return np.array([[hxx, hxy], [hxy, hyy]])


def himmelblau_der(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b = x * x + y - 11, x + y * y - 7
    return np.array([4 * x * a + 2 * b, 2 * a + 4 * y * b])


def himmelblau_hess(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b = x * x + y - 11, x + y * y - 7
    return np.array([[4 * a + 8 * x * x + 2, 4 * x + 4 * y], [4 * x + 4 * y, 2 + 4 * b + 8 * y * y]])


def spd_quadratic(x: FloatArray) -> float:
    return float(0.5 * x @ QUADRATIC_A @ x - QUADRATIC_B @ x)


def spd_quadratic_der(x: FloatArray) -> FloatArray:
    return QUADRATIC_A @ x - QUADRATIC_B


def spd_quadratic_hess(x: FloatArray) -> FloatArray:
    return QUADRATIC_A.copy()


def powell(v: FloatArray) -> float:
    """Powell's singular function (minimum 0 at the origin, singular Hessian there)."""
    a, b, c, d = (float(t) for t in v)
    return (a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4


def powell_der(v: FloatArray) -> FloatArray:
    a, b, c, d = (float(t) for t in v)
    return np.array(
        [
            2 * (a + 10 * b) + 40 * (a - d) ** 3,
            20 * (a + 10 * b) + 4 * (b - 2 * c) ** 3,
            10 * (c - d) - 8 * (b - 2 * c) ** 3,
            -10 * (c - d) - 40 * (a - d) ** 3,
        ]
    )


def powell_hess(v: FloatArray) -> FloatArray:
    a, b, c, d = (float(t) for t in v)
    p = 12 * (b - 2 * c) ** 2
    q = 120 * (a - d) ** 2
    return np.array(
        [
            [2 + q, 20, 0, -q],
            [20, 200 + p, -2 * p, 0],
            [0, -2 * p, 10 + 4 * p, -10],
            [-q, 0, -10, 10 + q],
        ]
    )


def wood(v: FloatArray) -> float:
    """Wood's (Colville's) four-variable function (minimum 0 at ones)."""
    a, b, c, d = (float(t) for t in v)
    return (
        100 * (b - a * a) ** 2
        + (1 - a) ** 2
        + 90 * (d - c * c) ** 2
        + (1 - c) ** 2
        + 10.1 * ((b - 1) ** 2 + (d - 1) ** 2)
        + 19.8 * (b - 1) * (d - 1)
    )


def wood_der(v: FloatArray) -> FloatArray:
    a, b, c, d = (float(t) for t in v)
    return np.array(
        [
            -400 * a * (b - a * a) - 2 * (1 - a),
            200 * (b - a * a) + 20.2 * (b - 1) + 19.8 * (d - 1),
            -360 * c * (d - c * c) - 2 * (1 - c),
            180 * (d - c * c) + 20.2 * (d - 1) + 19.8 * (b - 1),
        ]
    )


def wood_hess(v: FloatArray) -> FloatArray:
    a, b, c, d = (float(t) for t in v)
    return np.array(
        [
            [1200 * a * a - 400 * b + 2, -400 * a, 0, 0],
            [-400 * a, 220.2, 0, 19.8],
            [0, 0, 1080 * c * c - 360 * d + 2, -360 * c],
            [0, 19.8, -360 * c, 200.2],
        ]
    )


SMOOTH: dict[str, Smooth] = {
    "rosen": Smooth(rosen, rosen_der, rosen_hess, np.array([-1.2, 1.0])),
    "rosen5": Smooth(rosen, rosen_der, rosen_hess, np.array([1.3, 0.7, 0.8, 1.9, 1.2])),
    "beale": Smooth(beale, beale_der, beale_hess, np.array([1.0, 1.0])),
    "himmelblau": Smooth(himmelblau, himmelblau_der, himmelblau_hess, np.array([0.0, 0.0])),
    "spdQuadratic": Smooth(spd_quadratic, spd_quadratic_der, spd_quadratic_hess, np.array([1.0, 1.0, 1.0])),
    "powell": Smooth(powell, powell_der, powell_hess, np.array([3.0, -1.0, 0.0, 1.0])),
    "wood": Smooth(wood, wood_der, wood_hess, np.array([-3.0, -1.0, -3.0, -1.0])),
}
"""The smooth problems with derivatives, by name (the TypeScript tests use the same names)."""


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


# ---------------------------------------------------------------------------------------------------------------
# Least-squares residuals (the TypeScript least-squares tests define the same)

type Residuals = Callable[[FloatArray], FloatArray]

LS_T: FloatArray = np.linspace(0.0, 4.0, 12)
"""The sample times of the exponential-decay fits."""
_g = np.random.default_rng(11)
LS_Y: FloatArray = 2.5 * np.exp(-1.3 * LS_T) + 0.4 + 0.02 * _g.normal(size=LS_T.size)
"""Noisy exponential decay observations."""
LS_Y_OUTLIERS: FloatArray = LS_Y.copy()
LS_Y_OUTLIERS[[3, 8]] += np.array([1.5, -1.0])


def exp_model(t: FloatArray, a: float, k: float, c: float) -> FloatArray:
    """a exp(-k t) + c."""
    return a * np.exp(-k * t) + c


def rosenbrock_residuals(x: FloatArray) -> FloatArray:
    return np.array([10 * (x[1] - x[0] * x[0]), 1 - x[0]])


def helical_residuals(x: FloatArray) -> FloatArray:
    theta = np.arctan2(x[1], x[0]) / (2 * np.pi)
    return np.array([10 * (x[2] - 10 * theta), 10 * (np.hypot(x[0], x[1]) - 1), x[2]])


def beale_residuals(x: FloatArray) -> FloatArray:
    return np.array([1.5 - x[0] * (1 - x[1]), 2.25 - x[0] * (1 - x[1] ** 2), 2.625 - x[0] * (1 - x[1] ** 3)])


def exponential_residuals(x: FloatArray) -> FloatArray:
    return exp_model(LS_T, float(x[0]), float(x[1]), float(x[2])) - LS_Y


def outlier_residuals(x: FloatArray) -> FloatArray:
    return exp_model(LS_T, float(x[0]), float(x[1]), float(x[2])) - LS_Y_OUTLIERS


RESIDUALS: dict[str, tuple[Residuals, FloatArray]] = {
    "rosenbrock": (rosenbrock_residuals, np.array([-1.2, 1.0])),
    "helical": (helical_residuals, np.array([-1.0, 0.0, 0.0])),
    "beale": (beale_residuals, np.array([1.0, 1.0])),
    "exponential": (exponential_residuals, np.array([1.0, 1.0, 0.0])),
    "outliers": (outlier_residuals, np.array([1.0, 1.0, 0.0])),
}
"""Residual functions and starts by name."""

BOUNDED_RESIDUALS: list[tuple[str, list[tuple[float, float]]]] = [
    ("rosenbrock", [(-2.0, 0.8), (-2.0, 2.0)]),
    ("beale", [(0.0, 2.5), (0.0, 0.4)]),
    ("exponential", [(0.0, 2.0), (0.0, 5.0), (0.5, 1.0)]),
]
"""The bounded least-squares setups."""


def ls_summary(r: so.OptimizeResult) -> FloatArray:
    """The solution and cost of a least_squares result."""
    return np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.cost)]])


# ---------------------------------------------------------------------------------------------------------------
# Constrained problems (the TypeScript constrained tests define the same)

type Fn = Callable[[FloatArray], float]
type Vec = Callable[[FloatArray], FloatArray]


def tutorial(x: FloatArray) -> float:
    return float((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2)


def tutorial_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] - 2 * x[1] + 2, -x[0] - 2 * x[1] + 6, -x[0] + 2 * x[1] + 2])


def hs071(x: FloatArray) -> float:
    return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])


def hs071_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] * x[1] * x[2] * x[3] - 25])


def hs071_eq(x: FloatArray) -> FloatArray:
    return np.array([np.sum(x * x) - 40])


def rosen_ineq(x: FloatArray) -> FloatArray:
    return np.array([1 - x[0] - 2 * x[1], 1 - x[0] ** 2 - x[1], 1 - x[0] ** 2 + x[1]])


def rosen_eq(x: FloatArray) -> FloatArray:
    return np.array([1 - 2 * x[0] - x[1]])


def sphere(x: FloatArray) -> float:
    return float(np.sum(x * x))


def sphere_eq(x: FloatArray) -> FloatArray:
    return np.array([np.sum(x) - 1])


def sphere_ineq(x: FloatArray) -> FloatArray:
    return np.array([x[0] - 0.5])


def disc(x: FloatArray) -> float:
    return float((x[0] - 2) ** 2 + (x[1] - 1) ** 2)


def disc_ineq(x: FloatArray) -> FloatArray:
    return np.array([1 - x[0] ** 2 - x[1] ** 2])


CONSTRAINED: dict[str, tuple[Fn, Vec | None, Vec | None, list[tuple[float, float]] | None, list[float]]] = {
    "tutorial": (tutorial, None, tutorial_ineq, [(0.0, np.inf), (0.0, np.inf)], [2.0, 0.0]),
    "hs071": (hs071, hs071_eq, hs071_ineq, [(1.0, 5.0)] * 4, [1.0, 5.0, 5.0, 1.0]),
    "rosen": (rosen, rosen_eq, rosen_ineq, [(0.0, 1.0), (-0.5, 2.0)], [0.5, 0.0]),
    "sphere": (sphere, sphere_eq, sphere_ineq, None, [1.0, 1.0, 1.0]),
    "disc": (disc, None, disc_ineq, None, [0.0, 0.0]),
}
"""Constrained problems by name: objective, equalities, inequalities (>= 0), bounds and start."""


def slsqp_solution(name: str) -> so.OptimizeResult:
    """scipy's SLSQP solution of a constrained problem, at a tight tolerance."""
    f, eq, ineq, bounds, x0 = CONSTRAINED[name]
    constraints: list[so.NonlinearConstraint] = []
    if eq is not None:
        constraints.append(so.NonlinearConstraint(eq, 0.0, 0.0))
    if ineq is not None:
        constraints.append(so.NonlinearConstraint(ineq, 0.0, np.inf))
    return so.minimize(f, np.array(x0), method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": 1e-12})
