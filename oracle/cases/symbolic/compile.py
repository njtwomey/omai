"""Compiled numeric functions against `sympy.lambdify` with numpy: values on arrays of points (real arithmetic, so
NaN outside the domain, as numpy)."""

import numpy as np
import scipy.special as sps
import sympy as sp
from omai_oracle.fixture import Case, case
from omai_oracle.symbolic import SYMBOLS, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__, "numpy": np.__version__}

x, y = SYMBOLS["x"], SYMBOLS["y"]

FUNCTIONS: list[str] = [
    "x**2 + sin(y)",
    "exp(-x**2/2)/sqrt(2*pi)",
    "log(1 + exp(x))",
    "atan2(y, x)",
    "sqrt(x) + Abs(y)",
    "Piecewise((x**2, x < 1), (2*x - 1, True))",
    "x**y",
    "tanh(x)*cosh(y)",
    "1/(x**2 + y**2 + 1)",
    "erf(x) + y",
    "sec(x) + acot(y)",
    "floor(x) + sign(y)",
]


def cases() -> list[Case]:
    xs = np.linspace(-1.5, 2.5, 9)
    ys = np.linspace(0.25, 3.0, 9)
    out: list[Case] = []
    for i, s in enumerate(FUNCTIONS):
        f = sp.lambdify((x, y), parse(s), modules=[{"erf": sps.erf}, "numpy"])
        with np.errstate(all="ignore"):
            v = np.asarray(f(xs, ys), dtype=np.float64) * np.ones_like(xs)
        out.append(case(f"compile/{i}", "compile", [s, xs, ys], v, rtol=1e-12, atol=1e-12))
    return out
