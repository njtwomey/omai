"""Limits, Riemann sums, Taylor polynomials, Newton's method, critical points and the mean value theorem against sympy,
scipy.integrate and scipy.optimize."""

from typing import cast

import numpy as np
import scipy
import sympy
from mdml_oracle.fixture import Case, case
from scipy.integrate import trapezoid
from scipy.optimize import newton

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sympy.__version__}

x = sympy.Symbol("x", real=True)

LIMITS: list[tuple[str, sympy.Expr, sympy.Expr, str]] = [
    ("sinc", sympy.sin(x) / x, sympy.Integer(0), "+-"),
    ("compound", (1 + x) ** (1 / x), sympy.Integer(0), "+-"),
    ("expm1", (sympy.exp(x) - 1) / x, sympy.Integer(0), "+-"),
    ("hole", (x**2 - 1) / (x - 1), sympy.Integer(1), "+-"),
    ("versine", (1 - sympy.cos(x)) / x**2, sympy.Integer(0), "+-"),
    ("signRight", sympy.Abs(x) / x, sympy.Integer(0), "+"),
    ("signLeft", sympy.Abs(x) / x, sympy.Integer(0), "-"),
    ("euler", (1 + 1 / x) ** x, sympy.oo, "-"),
    ("rational", (3 * x**2 + x) / (x**2 - 5), sympy.oo, "-"),
]

FUNCTIONS: dict[str, sympy.Expr] = {
    "cubic": x**3 - 3 * x,
    "quartic": x**4 - 2 * x**2 + x / 2,
    "wave": sympy.sin(x) + x / 3,
    "expsin": sympy.exp(sympy.sin(x)),
    "log1p": sympy.log(1 + x),
    "gauss": sympy.exp(-(x**2)),
}


def diff(expr: sympy.Expr, k: int = 1) -> sympy.Expr:
    return sympy.diff(expr, x, k)


def real_roots(expr: sympy.Expr) -> list[float]:
    """The real roots of a polynomial in x, ascending."""
    return sorted(value(r) for r in cast(list[sympy.Basic], sympy.Poly(expr, x).real_roots()))


def value(expr: sympy.Basic) -> float:
    return float(sympy.N(expr, 20))


def riemann(f: sympy.Expr, a: float, b: float, n: int, rule: str) -> float:
    g = sympy.lambdify(x, f, "numpy")
    edges = np.linspace(a, b, n + 1)
    dx = (b - a) / n
    if rule == "left":
        return float(np.sum(g(edges[:-1])) * dx)
    if rule == "right":
        return float(np.sum(g(edges[1:])) * dx)
    if rule == "midpoint":
        return float(np.sum(g((edges[:-1] + edges[1:]) / 2)) * dx)
    return float(trapezoid(g(edges), edges))


def cases() -> list[Case]:
    out: list[Case] = []
    for name, expr, at, side in LIMITS:
        want = value(sympy.limit(expr, x, at, dir=side))
        point = np.array(float(at)) if at != sympy.oo else np.array(np.inf)
        out.append(case(f"limit/{name}", "limit", [point], np.array(want), rtol=1e-8, atol=1e-10, f=name, side=side))
    for name in ["cubic", "wave", "gauss"]:
        for rule in ["left", "right", "midpoint", "trapezoid"]:
            want = riemann(FUNCTIONS[name], -1.0, 2.0, 7, rule)
            out.append(case(f"riemann/{name}/{rule}", "riemann", [], np.array(want), rtol=1e-13, f=name, rule=rule))
    for name, centre in [("expsin", 0.5), ("log1p", 0.0), ("gauss", 1.0), ("wave", -0.3)]:
        # The Taylor coefficients f^(k)(c) / k! from sympy's symbolic derivatives.
        coeffs = [value(diff(FUNCTIONS[name], k).subs(x, centre)) / float(sympy.factorial(k)) for k in range(6)]
        out.append(
            case(f"taylor/{name}", "taylor", [np.array(centre)], np.array(coeffs), rtol=1e-11, atol=1e-13, f=name)
        )
        for k in range(1, 4):
            d = value(diff(FUNCTIONS[name], k).subs(x, centre))
            out.append(
                case(f"derivative/{name}/{k}", "derivative", [np.array(centre)], np.array(d), rtol=1e-11, f=name, k=k)
            )
    for name, x0 in [("cubic", 2.0), ("wave", -1.0), ("quartic", 1.5)]:
        f = sympy.lambdify(x, FUNCTIONS[name], "math")
        df = sympy.lambdify(x, diff(FUNCTIONS[name]), "math")
        root = float(newton(f, x0, fprime=df, tol=1e-14, maxiter=100))
        out.append(case(f"newton/{name}", "newton", [np.array(x0)], np.array(root), rtol=1e-12, f=name))
    for name in ["cubic", "quartic"]:
        d1 = diff(FUNCTIONS[name])
        roots = real_roots(d1)
        out.append(case(f"critical/{name}", "critical", [], np.array(roots), rtol=1e-10, atol=1e-12, f=name))
        kinds = ["minimum" if value(diff(d1).subs(x, r)) > 0 else "maximum" for r in roots]
        out.append(case(f"critical/{name}/kinds", "criticalKinds", [], ",".join(kinds), f=name))
    for name, a, b in [("cubic", -2.0, 2.5), ("quartic", -1.5, 1.2)]:
        fx = FUNCTIONS[name]
        slope = (fx.subs(x, b) - fx.subs(x, a)) / (b - a)
        points = [r for r in real_roots(diff(fx) - slope) if a < r < b]
        out.append(case(f"mvt/{name}", "meanValue", [np.array([a, b])], np.array(points), rtol=1e-10, f=name))
    return out
