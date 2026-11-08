"""SDE schemes driven by given Brownian increments against sdeint (itoEuler and stratHeun accept dW): a nonlinear
two-dimensional system with general (full-matrix) noise and with diagonal noise. Same increments, so the paths agree
to rounding."""

from collections.abc import Callable

import numpy as np
import sdeint
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"sdeint": "0.3.0", "numpy": np.__version__}

type Field = Callable[[FloatArray, float], FloatArray]


def drift(y: FloatArray, t: float) -> FloatArray:
    """f(y, t) = [-y0 + sin(y1), -0.5 y1 + 0.1 y0 y1 + 0.2 t]."""
    return np.array([-y[0] + np.sin(y[1]), -0.5 * y[1] + 0.1 * y[0] * y[1] + 0.2 * t])


def general(y: FloatArray, t: float) -> FloatArray:
    """A full 2 x 2 noise matrix depending on the state."""
    return np.array([[0.3 * y[0], 0.1], [0.05, 0.2 * np.cos(y[1])]])


def diagonal(y: FloatArray, t: float) -> FloatArray:
    """Diagonal noise: g_i depends on y_i only."""
    return np.diag([0.3 * y[0], 0.2 * np.cos(y[1])])


def cases() -> list[Case]:
    g = rng(2311)
    out: list[Case] = []
    tspan = np.linspace(0.0, 1.0, 21)
    y0 = np.array([1.0, 0.5])
    dw = g.normal(scale=np.sqrt(tspan[1] - tspan[0]), size=(20, 2))
    schemes: list[tuple[str, Callable[..., FloatArray]]] = [("euler", sdeint.itoEuler), ("heun", sdeint.stratHeun)]
    noises: list[tuple[str, Field]] = [("general", general), ("diagonal", diagonal)]
    for method, solver in schemes:
        for noise, field in noises:
            ys = solver(drift, field, y0, tspan, dW=dw)
            out.append(
                case(
                    f"{method}/{noise}",
                    "sdeint",
                    [y0, dw, tspan],
                    np.asarray(ys, dtype=np.float64),
                    rtol=1e-12,
                    atol=1e-12,
                    method=method,
                    noise=noise,
                )
            )
    return out
