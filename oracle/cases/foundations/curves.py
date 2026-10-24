"""Curves: arc length, curvature, torsion and Frenet frames (sympy derivatives, scipy.integrate.quad), conic sections
(sympy.geometry) and Bezier curves (scipy.interpolate.BPoly)."""

from collections.abc import Callable
from typing import cast

import numpy as np
import scipy
import sympy
from scipy.integrate import quad
from scipy.interpolate import BPoly
from sympy.geometry import Ellipse, Line, Parabola, Point, Point2D
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sympy.__version__}

t = sympy.Symbol("t", real=True)

CURVES: dict[str, list[sympy.Expr]] = {
    "helix": [2 * sympy.cos(t), 2 * sympy.sin(t), t / 2],
    "twisted": [t, t**2, t**3],
    "trefoil": [
        sympy.sin(t) + 2 * sympy.sin(2 * t),
        sympy.cos(t) - 2 * sympy.cos(2 * t),
        -sympy.sin(3 * t),
    ],
    "ellipse": [3 * sympy.cos(t), sympy.sin(t)],
    "cardioid": [(1 - sympy.cos(t)) * sympy.cos(t), (1 - sympy.cos(t)) * sympy.sin(t)],
}


def numeric(exprs: list[sympy.Expr]) -> Callable[[float], FloatArray]:
    """A function of t returning the vector of sympy expressions as floats."""
    f = sympy.lambdify(t, exprs, "numpy")

    def at(x: float) -> FloatArray:
        return np.asarray(f(x), dtype=np.float64)

    return at


def geometry(exprs: list[sympy.Expr], ts: FloatArray) -> dict[str, FloatArray]:
    """Curvature, torsion, tangent, normal and binormal from sympy's symbolic derivatives, evaluated at ts."""
    d1, d2, d3 = (numeric([sympy.diff(e, t, k) for e in exprs]) for k in (1, 2, 3))
    kappa: list[float] = []
    tau: list[float] = []
    tangent: list[FloatArray] = []
    normal: list[FloatArray] = []
    for x in ts:
        a, b, c = d1(float(x)), d2(float(x)), d3(float(x))
        speed = float(np.linalg.norm(a))
        tangent.append(a / speed)
        if a.size == 2:
            kappa.append(float(a[0] * b[1] - a[1] * b[0]) / speed**3)
            normal.append(np.array([-a[1], a[0]]) / speed)
        else:
            cr = np.cross(a, b)
            kappa.append(float(np.linalg.norm(cr)) / speed**3)
            tau.append(float(cr @ c) / float(cr @ cr))
            binormal = cr / np.linalg.norm(cr)
            normal.append(np.cross(binormal, a / speed))
    return {
        "curvature": np.array(kappa),
        "torsion": np.array(tau),
        "tangent": np.array(tangent),
        "normal": np.array(normal),
    }


def coefficients(expr: sympy.Expr) -> FloatArray:
    """[A, B, C, D, E, F] of a quadratic in x and y."""
    x, y = sympy.symbols("x y", real=True)
    p = sympy.Poly(sympy.expand(expr), x, y)
    monomials = [(2, 0), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0)]
    return np.array([float(p.coeff_monomial(x**i * y**j)) for i, j in monomials], dtype=np.float64)


def rotated(expr: sympy.Expr, angle: float, shift: tuple[float, float]) -> sympy.Expr:
    """The equation of a conic rotated by `angle` about the origin and then translated by `shift`."""
    x, y = sympy.symbols("x y", real=True)
    c, s = sympy.cos(angle), sympy.sin(angle)
    u = (x - shift[0]) * c + (y - shift[1]) * s
    v = -(x - shift[0]) * s + (y - shift[1]) * c
    return sympy.sympify(expr).subs({x: u, y: v}, simultaneous=True)


def place(p: Point, angle: float, shift: tuple[float, float]) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    q = cast(Point2D, p)
    px, py = float(sympy.N(q.x, 20)), float(sympy.N(q.y, 20))
    return np.array([px * c - py * s + shift[0], px * s + py * c + shift[1]])


def cases() -> list[Case]:
    g = rng(1205)
    out: list[Case] = []
    for name, exprs in CURVES.items():
        ts = np.sort(g.uniform(0.2, 2.8, size=4))
        geo = geometry(exprs, ts)
        speed = numeric([sympy.sqrt(sum(sympy.diff(e, t) ** 2 for e in exprs))])

        def integrand(x: float, speed: Callable[[float], FloatArray] = speed) -> float:
            return float(speed(x)[0])

        length, _ = quad(integrand, 0.0, 2.0, epsabs=1e-13, epsrel=1e-13)
        out.append(case(f"{name}/arcLength", "arcLength", [], np.array(length), rtol=1e-8, curve=name))
        out.append(case(f"{name}/curvature", "curvature", [ts], geo["curvature"], rtol=1e-10, curve=name))
        out.append(case(f"{name}/tangent", "tangent", [ts], geo["tangent"], rtol=1e-10, atol=1e-12, curve=name))
        out.append(case(f"{name}/normal", "normal", [ts], geo["normal"], rtol=1e-10, atol=1e-12, curve=name))
        if len(exprs) == 3:
            out.append(case(f"{name}/torsion", "torsion", [ts], geo["torsion"], rtol=1e-9, curve=name))

    x, y = sympy.symbols("x y", real=True)
    for i, (a, b, angle, shift) in enumerate(
        [(5, 3, 0.0, (0.0, 0.0)), (4, 1, 0.6, (1.0, -2.0)), (2, 1.5, -1.1, (3.0, 0.5))]
    ):
        e = cast(Ellipse, Ellipse(Point(0, 0), hradius=sympy.nsimplify(a), vradius=sympy.nsimplify(b)))
        eq = rotated(e.equation(x, y), angle, shift)
        coef = coefficients(eq)
        foci = np.array([place(f, angle, shift) for f in cast(tuple[Point, Point], e.foci)])
        order = np.lexsort((foci[:, 1], foci[:, 0]))
        out.append(case(f"ellipse/{i}/foci", "conicFoci", [coef], foci[order], rtol=1e-9, atol=1e-11))
        out.append(case(f"ellipse/{i}/eccentricity", "conicEccentricity", [coef], np.array(float(e.eccentricity))))
        out.append(case(f"ellipse/{i}/centre", "conicCentre", [coef], np.array(shift), rtol=1e-10, atol=1e-12))
        out.append(case(f"ellipse/{i}/type", "conicType", [coef], "ellipse"))
    for i, (a, b, angle, shift) in enumerate([(3, 4, 0.0, (0.0, 0.0)), (1, 2, 0.4, (-1.0, 2.0))]):
        # x^2 / a^2 - y^2 / b^2 = 1, with foci at +-sqrt(a^2 + b^2) on the transverse axis.
        eq = rotated(x**2 / a**2 - y**2 / b**2 - 1, angle, shift)
        coef = coefficients(eq)
        c = float(np.hypot(a, b))
        foci = np.array([place(Point(-c, 0), angle, shift), place(Point(c, 0), angle, shift)])
        order = np.lexsort((foci[:, 1], foci[:, 0]))
        out.append(case(f"hyperbola/{i}/foci", "conicFoci", [coef], foci[order], rtol=1e-9, atol=1e-11))
        out.append(case(f"hyperbola/{i}/eccentricity", "conicEccentricity", [coef], np.array(c / a), rtol=1e-10))
        out.append(case(f"hyperbola/{i}/type", "conicType", [coef], "hyperbola"))
    for i, (focus, directrix) in enumerate(
        [
            ((0, 1), Line(Point(0, -1), Point(1, -1))),
            ((2, 0), Line(Point(-1, 0), Point(-1, 1))),
            ((1, 1), Line(Point(0, -2), Point(2, 0))),
        ]
    ):
        par = Parabola(Point(*focus), directrix)
        coef = coefficients(par.equation(x, y))
        out.append(
            case(f"parabola/{i}/foci", "conicFoci", [coef], np.array([focus], dtype=np.float64), rtol=1e-9, atol=1e-11)
        )
        vertex = cast(Point2D, par.vertex)
        out.append(
            case(
                f"parabola/{i}/vertex",
                "conicVertex",
                [coef],
                np.array([float(sympy.N(vertex.x, 20)), float(sympy.N(vertex.y, 20))]),
                atol=1e-11,
            )
        )
        out.append(case(f"parabola/{i}/p", "conicFocalParameter", [coef], np.array(abs(float(par.p_parameter)))))
        out.append(case(f"parabola/{i}/type", "conicType", [coef], "parabola"))
    for i, (coef, kind) in enumerate(
        [
            ([1.0, 0, 1, 0, 0, 0], "point"),
            ([1.0, 0, -1, 0, 0, 0], "intersecting lines"),
            ([1.0, 0, 0, 0, 0, -1], "parallel lines"),
            ([1.0, 0, 1, 0, 0, 1], "imaginary ellipse"),
            ([1.0, 0, 1, -2, 0, -3], "circle"),
        ]
    ):
        out.append(case(f"degenerate/{i}", "conicType", [np.array(coef)], kind))

    for k in [2, 3, 4]:
        control = g.normal(size=(k + 1, 2))
        ts = np.linspace(0, 1, 7)
        curve = BPoly(control[:, None, :], [0.0, 1.0])
        out.append(case(f"bezier/{k}", "bezier", [control, ts], curve(ts), rtol=1e-12, atol=1e-14))
        out.append(
            case(
                f"bezier/{k}/derivative",
                "bezierDerivative",
                [control, ts],
                curve.derivative()(ts),
                rtol=1e-12,
                atol=1e-13,
            )
        )
    return out
