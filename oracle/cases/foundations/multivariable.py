"""Gradients, Jacobians, Hessians, Lagrange multipliers, divergence, curl, and line and surface integrals against sympy
(symbolic differentiation, solve and integrate)."""

from typing import cast

import numpy as np
import scipy
import sympy
from scipy.integrate import dblquad
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sympy.__version__}

x, y, z, t, u, v = sympy.symbols("x y z t u v", real=True)
XY = [x, y]
XYZ = [x, y, z]

SCALARS: dict[str, tuple[sympy.Expr, list[sympy.Symbol]]] = {
    "bowl": (x**2 + 3 * y**2, XY),
    "rosenbrock": ((1 - x) ** 2 + 100 * (y - x**2) ** 2, XY),
    "wave": (sympy.sin(x) * sympy.cos(y) + x * y / 4, XY),
    "gauss3": (sympy.exp(-(x**2) - 2 * y**2 - z**2 / 2) + x * z, XYZ),
}

FIELDS: dict[str, tuple[list[sympy.Expr], list[sympy.Symbol]]] = {
    "swirl": ([-y + x * z, x + y * z, z**2 - x * y], XYZ),
    "radial": ([x * (x**2 + y**2 + z**2), y, z * sympy.exp(x)], XYZ),
    "plane": ([x**2 - y, x * y**2], XY),
}


def value(expr: sympy.Basic) -> float:
    return float(sympy.N(expr, 20))


def at(expr: sympy.Expr, symbols: list[sympy.Symbol], point: FloatArray) -> float:
    return value(expr.subs(dict(zip(symbols, [float(p) for p in point], strict=True))))


def diff(expr: sympy.Expr, s: sympy.Symbol) -> sympy.Expr:
    return sympy.diff(expr, s)


def cases() -> list[Case]:
    g = rng(1207)
    out: list[Case] = []
    for name, (f, sym) in SCALARS.items():
        pts = g.uniform(-1, 1, size=(3, len(sym)))
        grads = np.array([[at(diff(f, s), sym, p) for s in sym] for p in pts])
        hess = np.array([[[at(diff(diff(f, a), b), sym, p) for b in sym] for a in sym] for p in pts])
        out.append(case(f"gradient/{name}", "gradient", [pts], grads, rtol=1e-12, atol=1e-13, f=name))
        out.append(case(f"hessian/{name}", "hessian", [pts], hess, rtol=1e-12, atol=1e-13, f=name))
        direction = g.normal(size=len(sym))
        unit = direction / np.linalg.norm(direction)
        out.append(case(f"directional/{name}", "directional", [pts, direction], grads @ unit, atol=1e-13, f=name))
    for name, (field, sym) in FIELDS.items():
        pts = g.uniform(-1, 1, size=(3, len(sym)))
        jac = np.array([[[at(diff(c, s), sym, p) for s in sym] for c in field] for p in pts])
        out.append(case(f"jacobian/{name}", "jacobian", [pts], jac, rtol=1e-12, atol=1e-13, field=name))
        div = np.array([sum(at(diff(c, s), sym, p) for c, s in zip(field, sym, strict=True)) for p in pts])
        out.append(case(f"divergence/{name}", "divergence", [pts], div, rtol=1e-12, atol=1e-13, field=name))
        if len(sym) == 3:
            fx, fy, fz = field
            curl = [diff(fz, y) - diff(fy, z), diff(fx, z) - diff(fz, x), diff(fy, x) - diff(fx, y)]
            want = np.array([[at(c, sym, p) for c in curl] for p in pts])
        else:
            fx, fy = field
            want = np.array([at(diff(fy, x) - diff(fx, y), sym, p) for p in pts])
        out.append(case(f"curl/{name}", "curl", [pts], want, rtol=1e-12, atol=1e-13, field=name))

    lam = sympy.Symbol("lam", real=True)
    problems: list[tuple[str, sympy.Expr, sympy.Expr, list[float]]] = [
        ("xyCircle", x * y, x**2 + y**2 - 2, [1.2, 0.8]),
        ("sumEllipse", x + 2 * y, x**2 + 4 * y**2 - 8, [1.5, 1.0]),
        ("distanceLine", (x - 3) ** 2 + (y - 1) ** 2, x + y - 1, [0.0, 0.0]),
    ]
    for name, f, gc, start in problems:
        eqs = [diff(f, x) - lam * diff(gc, x), diff(f, y) - lam * diff(gc, y), gc]
        sols = cast(list[dict[sympy.Symbol, sympy.Expr]], sympy.solve(eqs, [x, y, lam], dict=True))
        # The stationary point nearest the start (where Newton from the start converges).
        best = min(sols, key=lambda s: (value(s[x]) - start[0]) ** 2 + (value(s[y]) - start[1]) ** 2)
        want = np.array([value(best[x]), value(best[y]), value(best[lam])])
        out.append(case(f"lagrange/{name}", "lagrange", [np.array(start)], want, rtol=1e-9, atol=1e-11, problem=name))

    # Line integrals along a helix and a parabola, surface integrals over a paraboloid patch.
    helix = [sympy.cos(t), sympy.sin(t), t / 2]
    speed = sympy.sqrt(sum(diff(c, t) ** 2 for c in helix))
    density = x**2 + z
    scalar = sympy.integrate(sympy.simplify(density.subs(dict(zip(XYZ, helix, strict=True))) * speed), (t, 0, 2))
    out.append(case("line/scalar", "scalarLine", [], np.array(value(scalar)), rtol=1e-10))
    swirl = FIELDS["swirl"][0]
    work = sympy.integrate(
        sum(c.subs(dict(zip(XYZ, helix, strict=True))) * diff(h, t) for c, h in zip(swirl, helix, strict=True)),
        (t, 0, 2),
    )
    out.append(case("line/vector", "vectorLine", [], np.array(value(work)), rtol=1e-10))
    patch = [u, v, u**2 + v**2]
    ru = sympy.Matrix([diff(c, u) for c in patch])
    rv = sympy.Matrix([diff(c, v) for c in patch])
    normal = ru.cross(rv)
    sub = dict(zip(XYZ, patch, strict=True))
    area_element = sympy.sqrt(normal.dot(normal))
    integrand = sympy.lambdify((v, u), (x * y + z).subs(sub) * area_element, "math")

    def element(b: float, a: float) -> float:
        return float(integrand(b, a))

    scalar_surface, _ = dblquad(element, 0.0, 1.0, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)
    out.append(case("surface/scalar", "scalarSurface", [], np.array(scalar_surface), rtol=1e-10))
    flux = sympy.integrate(sum(c.subs(sub) * n for c, n in zip(swirl, normal, strict=True)), (u, 0, 1), (v, 0, 1))
    out.append(case("surface/flux", "surfaceFlux", [], np.array(value(flux)), rtol=1e-10))
    return out
