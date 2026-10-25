"""Finite differences against numpy.gradient and scipy.differentiate (derivative, jacobian, hessian)."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy
import scipy.differentiate as sd
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Scalar = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

# name -> (f, point); the TypeScript test defines the same functions.
FUNCTIONS: dict[str, tuple[Scalar, float]] = {
    "exp": (np.exp, 1.0),
    "sin": (np.sin, 0.7),
    "log": (np.log, 2.5),
    "atan": (np.arctan, -0.3),
    "poly": (lambda x: x**5 - 3 * x**2 + 1, 1.3),
    "gaussian": (lambda x: np.exp(-(x**2)), 0.4),
}


def vector_function(x: FloatArray) -> FloatArray:
    """F(x) = [x0^2 x1, 5 x0 + sin x1, exp(x0 x2)] (columns of points are evaluated at once)."""
    return np.stack([x[0] ** 2 * x[1], 5 * x[0] + np.sin(x[1]), np.exp(x[0] * x[2])])


def rosenbrock(x: FloatArray) -> FloatArray:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[1]) ** 2 + 100 * (x[2] - x[1] ** 2) ** 2


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (f, x) in FUNCTIONS.items():
        res = sd.derivative(f, np.array(x))
        out.append(case(f"derivative/{name}", "derivative", [], np.array(res.df), rtol=1e-9, atol=1e-11, fn_name=name))
    g = rng(41)
    y = g.normal(size=9)
    xs = np.sort(g.uniform(0, 4, size=9))
    for edge in (1, 2):
        out.append(
            case(
                f"gradient/uniform/{edge}",
                "gradient",
                [y],
                np.gradient(y, 0.5, edge_order=edge),
                rtol=1e-12,
                edgeOrder=edge,
                spacing=0.5,
            )
        )
        out.append(
            case(
                f"gradient/coordinates/{edge}",
                "gradient",
                [y, xs],
                np.gradient(y, xs, edge_order=edge),
                rtol=1e-11,
                edgeOrder=edge,
            )
        )
    y2 = g.normal(size=(4, 5))
    for axis in (0, 1):
        out.append(
            case(
                f"gradient/2d/axis{axis}",
                "gradient",
                [y2],
                np.gradient(y2, axis=axis, edge_order=2),
                rtol=1e-12,
                axis=axis,
                edgeOrder=2,
            )
        )
    point = np.array([0.8, -0.4, 0.6])
    out.append(
        case("jacobian/vector", "jacobian", [point], sd.jacobian(vector_function, point).df, rtol=1e-7, atol=1e-8)
    )
    out.append(case("hessian/rosenbrock", "hessian", [point], sd.hessian(rosenbrock, point).ddf, rtol=1e-5, atol=1e-4))
    return out
