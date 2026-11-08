"""Elementwise math functions against numpy and scipy."""

import numpy as np
import scipy.special as sp
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import UNARY_SHAPES, FloatArray, Unary, rng, softplus, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def relu(x: FloatArray) -> FloatArray:
    return np.maximum(x, 0)


DOMAINS: dict[str, tuple[Unary, float, float]] = {
    "exp": (np.exp, -5, 5),
    "expm1": (np.expm1, -1e-3, 1e-3),
    "log": (np.log, 0.01, 10),
    "log1p": (np.log1p, -0.5, 1e-3),
    "log2": (np.log2, 0.01, 10),
    "log10": (np.log10, 0.01, 10),
    "sqrt": (np.sqrt, 0, 10),
    "sin": (np.sin, -10, 10),
    "cos": (np.cos, -10, 10),
    "tan": (np.tan, -1.5, 1.5),
    "arcsin": (np.arcsin, -1, 1),
    "arccos": (np.arccos, -1, 1),
    "arctan": (np.arctan, -10, 10),
    "sinh": (np.sinh, -5, 5),
    "cosh": (np.cosh, -5, 5),
    "tanh": (np.tanh, -5, 5),
    "arcsinh": (np.arcsinh, -10, 10),
    "arccosh": (np.arccosh, 1, 10),
    "arctanh": (np.arctanh, -0.99, 0.99),
    "sigmoid": (sp.expit, -40, 40),
    "softplus": (softplus, -40, 40),
    "relu": (relu, -2, 2),
    "floor": (np.floor, -5, 5),
    "ceil": (np.ceil, -5, 5),
    "trunc": (np.trunc, -5, 5),
}


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, (fn, lo, hi)) in enumerate(DOMAINS.items()):
        for shape in UNARY_SHAPES:
            x = uniform(rng(100 + i), shape, lo, hi)
            out.append(case(f"{name}/{list(shape)}", name, [x], fn(x), rtol=1e-13, atol=1e-15))
    halves = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    out.append(case("round/halves-to-even", "round", [halves], np.round(halves)))
    extremes = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    out.append(case("sigmoid/extremes", "sigmoid", [extremes], sp.expit(extremes)))
    out.append(case("softplus/extremes", "softplus", [extremes], np.logaddexp(0, extremes)))
    return out
