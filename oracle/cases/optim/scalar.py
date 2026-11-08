"""Scalar minimisation against scipy.optimize: minimize_scalar (brent, golden, bounded) and bracket.

The algorithms are deterministic ports, so iteration and evaluation counts are compared as well as the minimiser.
"""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Scalar = Callable[[float], float]


def shifted(x: float) -> float:
    return (x - 2) ** 2 + 1


def cosh1(x: float) -> float:
    return float(np.cosh(x - 1))


def quartic(x: float) -> float:
    return x**4 - 3 * x**3 + 2


def expline(x: float) -> float:
    return float(np.exp(x) - 2 * x)


def kink(x: float) -> float:
    return abs(x - 0.3) + 0.1 * x * x


def wave(x: float) -> float:
    return float(np.sin(x) + 0.1 * x)


FUNCTIONS: dict[str, Scalar] = {
    "shifted": shifted,
    "cosh1": cosh1,
    "quartic": quartic,
    "expline": expline,
    "kink": kink,
    "wave": wave,
}

# (function, bracket for brent and golden, bounds)
SETUPS: list[tuple[str, tuple[float, ...] | None, tuple[float, float]]] = [
    ("shifted", None, (0.0, 5.0)),
    ("cosh1", (0.0, 0.5), (-3.0, 3.0)),
    ("quartic", (1.0, 2.0, 4.0), (1.0, 4.0)),
    ("expline", None, (-1.0, 0.5)),
    ("kink", (-1.0, 0.0), (-2.0, 2.0)),
    ("wave", (3.0, 4.0, 6.0), (2.0, 7.0)),
]


def summary(r: so.OptimizeResult) -> list[float]:
    return [float(r.x), float(r.fun), float(r.nit), float(r.nfev)]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, br, bounds in SETUPS:
        f = FUNCTIONS[name]
        extra: dict[str, object] = {"function": name}
        if br is not None:
            extra["bracket"] = list(br)
        for method in ("brent", "golden"):
            r = so.minimize_scalar(f, bracket=br, method=method)
            out.append(case(f"{method}/{name}", method, [], np.array(summary(r)), rtol=1e-12, atol=1e-14, **extra))
        r = so.minimize_scalar(f, bounds=bounds, method="bounded")
        out.append(
            case(
                f"bounded/{name}",
                "bounded",
                [],
                np.array(summary(r)),
                rtol=1e-12,
                atol=1e-14,
                function=name,
                bounds=list(bounds),
            )
        )
        r = so.minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-9})
        out.append(
            case(
                f"bounded-tight/{name}",
                "bounded",
                [],
                np.array(summary(r)),
                rtol=1e-12,
                atol=1e-14,
                function=name,
                bounds=list(bounds),
                tol=1e-9,
            )
        )
        xa, xb = (br[0], br[1]) if br is not None and len(br) == 2 else (0.0, 1.0)
        b = so.bracket(f, xa=xa, xb=xb)
        out.append(
            case(
                f"bracket/{name}",
                "bracket",
                [],
                np.array([float(v) for v in b]),
                rtol=1e-12,
                atol=1e-14,
                function=name,
                xa=xa,
                xb=xb,
            )
        )
    return out
