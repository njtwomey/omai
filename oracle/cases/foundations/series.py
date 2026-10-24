"""Sequences, series, convergence tests, Fourier series, the Gibbs phenomenon and generating functions against sympy
(summation, limits, integrals, series), numpy and scipy.optimize."""

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy
import sympy
from scipy.optimize import minimize_scalar
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}

n = sympy.Symbol("n", integer=True, positive=True)
k = sympy.Symbol("k", integer=True, nonnegative=True)
x = sympy.Symbol("x", real=True)

TERMS: dict[str, sympy.Expr] = {
    "basel": 1 / (k + 1) ** 2,
    "alternatingHarmonic": (-1) ** k / (k + 1),
    "leibniz": (-1) ** k / (2 * k + 1),
    "exp2": 2**k / sympy.factorial(k),
    "alternatingSquares": (-1) ** k / (k + 1) ** 2,
    "geometric": sympy.Rational(2, 3) ** k,
}

RATIOS: dict[str, sympy.Expr] = {
    "polyOverExp": (n**2 + 1) / 2**n,
    "expOverPoly": 3**n / n**4,
    "fifthOverThird": n**5 / 3**n,
}

ROOTS: dict[str, sympy.Expr] = {"half": (n / (2 * n + 1)) ** n, "third": (n / (3 * n + 2)) ** n}

RADII: dict[str, sympy.Expr] = {"two": (n + 1) / 2**n, "five": n**2 / 5**n, "three": 1 / (n * 3**n)}

FOURIER: dict[str, sympy.Expr] = {"ramp": x, "parabola": x**2, "abs": sympy.Abs(x), "mixed": x + sympy.cos(2 * x)}


def value(expr: sympy.Basic) -> float:
    return float(sympy.N(expr, 20))


def coefficient(f: sympy.Expr, kind: str, m: int) -> float:
    """(1/pi) int_{-pi}^{pi} f cos(m x) or sin(m x) dx, split at 0 for |x|."""
    w = sympy.cos(m * x) if kind == "a" else sympy.sin(m * x)
    if f.has(sympy.Abs):
        left = sympy.integrate(-x * w, (x, -sympy.pi, 0))
        right = sympy.integrate(x * w, (x, 0, sympy.pi))
        return value((left + right) / sympy.pi)
    return value(sympy.integrate(f * w, (x, -sympy.pi, sympy.pi)) / sympy.pi)


def square_partial(t: float, terms: int) -> float:
    ks = np.arange(1, terms + 1, 2)
    return float(np.sum(4 / (np.pi * ks) * np.sin(ks * t)))


def cases() -> list[Case]:
    out: list[Case] = []
    for name, term in TERMS.items():
        partial = [value(sympy.summation(term, (k, 0, m))) for m in range(12)]
        out.append(case(f"partial/{name}", "partialSums", [], np.array(partial), rtol=1e-13, terms=name))
        total = value(sympy.summation(term, (k, 0, sympy.oo)))
        out.append(case(f"sum/{name}", "seriesSum", [], np.array(total), rtol=1e-9, terms=name))
    for name, term in RATIOS.items():
        ratio = value(sympy.limit(sympy.Abs(term.subs(n, n + 1) / term), n, sympy.oo))
        out.append(case(f"ratio/{name}", "ratioTest", [], np.array(ratio), rtol=1e-7, terms=name))
    for name, term in ROOTS.items():
        root = value(sympy.limit(sympy.Abs(term) ** (1 / n), n, sympy.oo))
        out.append(case(f"root/{name}", "rootTest", [], np.array(root), rtol=1e-7, terms=name))
    for name, coef in RADII.items():
        radius = value(1 / sympy.limit(sympy.Abs(coef) ** (1 / n), n, sympy.oo))
        out.append(case(f"radius/{name}", "radius", [], np.array(radius), rtol=1e-7, coefficients=name))
    for name, f in FOURIER.items():
        a = [coefficient(f, "a", m) for m in range(6)]
        b = [0.0] + [coefficient(f, "b", m) for m in range(1, 6)]
        out.append(case(f"fourier/{name}/a", "fourierA", [], np.array(a), rtol=1e-10, atol=1e-12, f=name))
        out.append(case(f"fourier/{name}/b", "fourierB", [], np.array(b), rtol=1e-10, atol=1e-12, f=name))
    for terms in [9, 31, 101]:
        width = 4 * np.pi / (2 * terms + 1)

        def negative(t: float, terms: int = terms) -> float:
            return -square_partial(t, terms)

        res = minimize_scalar(negative, bounds=(1e-9, width), method="bounded", options={"xatol": 1e-13})
        peak = float(-res.fun)
        out.append(case(f"gibbs/{terms}", "gibbsPeak", [], np.array(peak), rtol=1e-10, terms=terms))
    rationals: list[tuple[str, list[int], list[int]]] = [
        ("fibonacci", [0, 1], [1, -1, -1]),
        ("catalanLike", [1, 2], [1, -3, 1]),
        ("tribonacci", [0, 0, 1], [1, -1, -1, -1]),
    ]
    for name, num, den in rationals:
        p = sum(c * x**i for i, c in enumerate(num))
        q = sum(c * x**i for i, c in enumerate(den))
        poly = sympy.Poly(cast(sympy.Expr, sympy.series(p / q, x, 0, 12)).removeO(), x)
        coeffs = [value(poly.coeff_monomial(x**i)) for i in range(12)]
        out.append(case(f"rational/{name}", "rationalSeries", [np.array(num), np.array(den)], np.array(coeffs)))
    a = np.array([1.0, -2.0, 0.5, 3.0])
    b = np.array([2.0, 1.0, -1.0])
    conv: npt.NDArray[np.float64] = np.convolve(a, b)
    out.append(case("product", "seriesProduct", [a, b], conv))
    return out
