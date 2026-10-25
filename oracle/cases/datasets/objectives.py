"""Optimisation test functions against scipy.optimize's Rosenbrock (value, gradient and Hessian) and closed forms."""

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.optimize import rosen, rosen_der, rosen_hess

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}


def himmelblau(p: FloatArray) -> float:
    x, y = float(p[0]), float(p[1])
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


def branin(p: FloatArray) -> float:
    x1, x2 = float(p[0]), float(p[1])
    b, c, t = 5.1 / (4 * np.pi**2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1 * x1 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(3)
    for d in [2, 4]:
        for i in range(3):
            x = g.uniform(-1.5, 1.5, size=d)
            out.append(case(f"rosenbrock/{d}/{i}/f", "rosenbrock", [x], np.asarray(rosen(x)), rtol=1e-12, part="f"))
            out.append(
                case(f"rosenbrock/{d}/{i}/grad", "rosenbrock", [x], np.asarray(rosen_der(x)), rtol=1e-12, part="g")
            )
            out.append(
                case(f"rosenbrock/{d}/{i}/hess", "rosenbrock", [x], np.asarray(rosen_hess(x)), rtol=1e-12, part="h")
            )
    for name, f in [("himmelblau", himmelblau), ("branin", branin)]:
        pts = g.uniform(0, 5, size=(4, 2))
        out.append(case(f"{name}/values", name, [pts], np.asarray([f(p) for p in pts]), rtol=1e-12, part="f"))
    return out
