"""Standard smooth test problems for the optim fixtures, with analytic gradients and Hessians.

The TypeScript tests define the same problems (packages/engine/src/testing/problems.ts) and differentiate them.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.optimize as so

from .helpers import FloatArray

QUADRATIC_A: FloatArray = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.5], [0.5, -0.5, 2.0]])
QUADRATIC_B: FloatArray = np.array([1.0, -2.0, 0.5])


@dataclass(frozen=True)
class Problem:
    """A test problem: objective, gradient, Hessian and the standard starting point."""

    f: Callable[[FloatArray], float]
    grad: Callable[[FloatArray], FloatArray]
    hess: Callable[[FloatArray], FloatArray]
    x0: FloatArray


def rosen(x: FloatArray) -> float:
    return float(so.rosen(x))


def rosen_der(x: FloatArray) -> FloatArray:
    return np.asarray(so.rosen_der(x), dtype=np.float64)


def rosen_hess(x: FloatArray) -> FloatArray:
    return np.asarray(so.rosen_hess(x), dtype=np.float64)


def beale(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2


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


def himmelblau(v: FloatArray) -> float:
    x, y = float(v[0]), float(v[1])
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


def himmelblau_der(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b = x * x + y - 11, x + y * y - 7
    return np.array([4 * x * a + 2 * b, 2 * a + 4 * y * b])


def himmelblau_hess(v: FloatArray) -> FloatArray:
    x, y = float(v[0]), float(v[1])
    a, b = x * x + y - 11, x + y * y - 7
    return np.array([[4 * a + 8 * x * x + 2, 4 * x + 4 * y], [4 * x + 4 * y, 2 + 4 * b + 8 * y * y]])


def quadratic(x: FloatArray) -> float:
    return float(0.5 * x @ QUADRATIC_A @ x - QUADRATIC_B @ x)


def quadratic_der(x: FloatArray) -> FloatArray:
    return QUADRATIC_A @ x - QUADRATIC_B


def quadratic_hess(x: FloatArray) -> FloatArray:
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


PROBLEMS: dict[str, Problem] = {
    "rosen": Problem(rosen, rosen_der, rosen_hess, np.array([-1.2, 1.0])),
    "rosen5": Problem(rosen, rosen_der, rosen_hess, np.array([1.3, 0.7, 0.8, 1.9, 1.2])),
    "beale": Problem(beale, beale_der, beale_hess, np.array([1.0, 1.0])),
    "himmelblau": Problem(himmelblau, himmelblau_der, himmelblau_hess, np.array([0.0, 0.0])),
    "quadratic": Problem(quadratic, quadratic_der, quadratic_hess, np.array([1.0, 1.0, 1.0])),
    "powell": Problem(powell, powell_der, powell_hess, np.array([3.0, -1.0, 0.0, 1.0])),
    "wood": Problem(wood, wood_der, wood_hess, np.array([-3.0, -1.0, -3.0, -1.0])),
}
"""The test problems by name, as the TypeScript tests know them."""
