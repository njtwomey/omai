"""Approximation against numpy.polynomial.chebyshev, scipy.interpolate.pade and numpy's floating-point functions."""

import math

import numpy as np
import numpy.polynomial.chebyshev as C
import scipy
import scipy.integrate as si
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def exp_pade(n: int, m: int) -> FloatArray:
    """The closed-form [n/m] Pade approximant of exp: numerator then denominator, lowest degree first."""
    f = math.factorial
    p = [f(n + m - k) * f(n) / (f(n + m) * f(k) * f(n - k)) for k in range(n + 1)]
    q = [(-1) ** k * f(n + m - k) * f(m) / (f(n + m) * f(k) * f(m - k)) for k in range(m + 1)]
    return np.array(p + q, dtype=np.float64)


def inside_roots(c: FloatArray) -> FloatArray:
    """Real roots of a Chebyshev series within [-1, 1], sorted."""
    raw = C.chebroots(c)
    return np.sort(np.real(raw[(np.abs(np.imag(raw)) < 1e-10) & (np.abs(np.real(raw)) <= 1)]))


def smooth(x: float) -> float:
    return float(np.exp(x) * np.sin(5 * x))


def cases() -> list[Case]:
    g = rng(120)
    out: list[Case] = []
    n = 9
    pts = np.cos(np.arange(n) * np.pi / (n - 1))
    vals = np.exp(pts) * np.cos(3 * pts)
    out.append(case("coefficients", "coefficients", [vals], C.chebfit(pts, vals, n - 1), rtol=1e-10, atol=1e-12))
    c = g.normal(size=7)
    out.append(case("derivative", "derivative", [c], C.chebder(c, 2), rtol=1e-12, atol=1e-12, order=2))
    out.append(case("cumsum", "cumsum", [c], C.chebint(c, lbnd=-1), rtol=1e-12, atol=1e-12))
    out.append(
        case(
            "roots",
            "roots",
            [c],
            inside_roots(c),
            rtol=1e-9,
            atol=1e-10,
        )
    )
    total, _ = si.quad(smooth, 0, 2, epsabs=1e-14, epsrel=1e-14)
    out.append(case("chebfun/sum", "chebfunSum", [], np.array(total), rtol=1e-12))
    xs = np.linspace(0, 2, 11)
    out.append(case("chebfun/values", "chebfunValues", [xs], np.exp(xs) * np.sin(5 * xs), rtol=1e-13, atol=1e-13))
    out.append(
        case("chebfun/roots", "chebfunRoots", [], np.array([k * np.pi / 5 for k in range(4)]), rtol=1e-12, atol=1e-13)
    )

    taylor = np.array([1 / math.factorial(k) for k in range(8)], dtype=np.float64)
    for m in (1, 2, 3):
        n = 7 - m
        out.append(case(f"pade/{m}", "pade", [taylor], exp_pade(n, m), rtol=1e-11, atol=1e-13, m=m))

    xf = np.array([1.0, -3.5, 1e-300, 7.25e10, 0.1])
    out.append(case("nextAfter/up", "nextAfterUp", [xf], np.nextafter(xf, np.inf), rtol=0))
    out.append(case("nextAfter/down", "nextAfterDown", [xf], np.nextafter(xf, -np.inf), rtol=0))
    out.append(case("ulp", "ulp", [xf], np.spacing(np.abs(xf)), rtol=0))
    out.append(case("float32", "float32", [xf], xf.astype(np.float32).astype(np.float64), rtol=0))
    # Ties, subnormals (the smallest is 2^-24), the largest finite value and rounding at the top of the range.
    xh = np.array(
        [
            0.1,
            -3.14159,
            1 + 2.0**-11,
            1 + 3 * 2.0**-11,
            2.0**-24 * 1.5,
            2.0**-25 * 1.01,
            6e-8,
            65504.0,
            65519.0,
            1e-9,
            -2049.0,
        ]
    )
    out.append(case("float16", "float16", [xh], xh.astype(np.float16).astype(np.float64), rtol=0))
    out.append(
        case(
            "epsilon",
            "epsilon",
            [],
            np.array([np.finfo(np.float64).eps, np.finfo(np.float32).eps], dtype=np.float64),
            rtol=0,
        )
    )
    return out
