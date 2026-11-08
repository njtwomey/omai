"""Differential geometry of surfaces against sympy (symbolic fundamental forms, curvatures and Christoffel symbols
from the metric) and scipy.integrate.solve_ivp (geodesics)."""

from collections.abc import Callable
from typing import cast

import numpy as np
import scipy
import sympy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.integrate import solve_ivp

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sympy.__version__}

u, v = sympy.symbols("u v", real=True)

SURFACES: dict[str, list[sympy.Expr]] = {
    "sphere": [2 * sympy.sin(u) * sympy.cos(v), 2 * sympy.sin(u) * sympy.sin(v), 2 * sympy.cos(u)],
    "torus": [(3 + sympy.cos(u)) * sympy.cos(v), (3 + sympy.cos(u)) * sympy.sin(v), sympy.sin(u)],
    "saddle": [u, v, u * v],
    "monkey": [u, v, u**3 - 3 * u * v**2],
    "catenoid": [sympy.cosh(u) * sympy.cos(v), sympy.cosh(u) * sympy.sin(v), u],
    "helicoid": [u * sympy.cos(v), u * sympy.sin(v), v / 2],
}

type Field = Callable[[float, float], float]


def lam(expr: sympy.Expr) -> Field:
    f = sympy.lambdify((u, v), expr, "math")

    def at(a: float, b: float) -> float:
        return float(f(a, b))

    return at


def derivative(m: sympy.Matrix, s: sympy.Symbol) -> sympy.Matrix:
    return cast(sympy.Matrix, m.diff(s))


def geometry(exprs: list[sympy.Expr]) -> dict[str, sympy.Expr]:
    r = sympy.Matrix(exprs)
    ru, rv = derivative(r, u), derivative(r, v)
    n = ru.cross(rv)
    n = n / sympy.sqrt(n.dot(n))
    e, f, g = ru.dot(ru), ru.dot(rv), rv.dot(rv)
    ll, m, nn = derivative(ru, u).dot(n), derivative(ru, v).dot(n), derivative(rv, v).dot(n)
    det = e * g - f**2
    k = (ll * nn - m**2) / det
    h = (e * nn - 2 * f * m + g * ll) / (2 * det)
    return {"E": e, "F": f, "G": g, "L": ll, "M": m, "N": nn, "K": k, "H": h}


def christoffel(exprs: list[sympy.Expr]) -> list[list[list[sympy.Expr]]]:
    """Gamma^k_ij = 1/2 g^{kl} (d_i g_jl + d_j g_il - d_l g_ij), from the metric alone."""
    r = sympy.Matrix(exprs)
    ru, rv = derivative(r, u), derivative(r, v)
    g = sympy.Matrix([[ru.dot(ru), ru.dot(rv)], [ru.dot(rv), rv.dot(rv)]])
    ginv = g.inv()
    x = [u, v]
    return [
        [
            [
                sum(
                    (
                        ginv[k, ll]
                        * (sympy.diff(g[j, ll], x[i]) + sympy.diff(g[i, ll], x[j]) - sympy.diff(g[i, j], x[ll]))
                        / 2
                        for ll in range(2)
                    ),
                    sympy.Integer(0),
                )
                for j in range(2)
            ]
            for i in range(2)
        ]
        for k in range(2)
    ]


def cases() -> list[Case]:
    g = rng(1206)
    out: list[Case] = []
    for name, exprs in SURFACES.items():
        geo = {key: lam(expr) for key, expr in geometry(exprs).items()}
        uv = np.stack([g.uniform(0.3, 1.2, size=4), g.uniform(-1.0, 1.0, size=4)], -1)
        for key, f in geo.items():
            want = np.array([f(float(a), float(b)) for a, b in uv])
            out.append(case(f"{name}/{key}", key, [uv], want, rtol=1e-10, atol=1e-12, surface=name))
        principal = []
        for a, b in uv:
            # Eigenvalues of the shape operator I^-1 II (numpy), well conditioned even at umbilics.
            first = np.array([[geo[k](float(a), float(b)) for k in row] for row in (("E", "F"), ("F", "G"))])
            second = np.array([[geo[k](float(a), float(b)) for k in row] for row in (("L", "M"), ("M", "N"))])
            ks = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(first, second))))[::-1]
            principal.append(ks)
        out.append(
            case(f"{name}/principal", "principal", [uv], np.array(principal), rtol=1e-8, atol=1e-10, surface=name)
        )
        gamma = christoffel(exprs)
        a, b = float(uv[0, 0]), float(uv[0, 1])
        want = np.array([[[lam(gamma[k][i][j])(a, b) for j in range(2)] for i in range(2)] for k in range(2)])
        out.append(case(f"{name}/christoffel", "christoffel", [uv[0]], want, rtol=1e-10, atol=1e-12, surface=name))

        fields = [[[lam(gamma[k][i][j]) for j in range(2)] for i in range(2)] for k in range(2)]

        def rhs(_: float, s: FloatArray, fields: list[list[list[Field]]] = fields) -> FloatArray:
            du = s[2:]
            acc = [
                -sum(fields[k][i][j](s[0], s[1]) * du[i] * du[j] for i in range(2) for j in range(2)) for k in range(2)
            ]
            return np.array([du[0], du[1], acc[0], acc[1]])

        e0, f0, g0 = (geo[key](a, b) for key in ("E", "F", "G"))
        vel = np.array([0.6, 0.8])
        speed = np.sqrt(e0 * vel[0] ** 2 + 2 * f0 * vel[0] * vel[1] + g0 * vel[1] ** 2)
        sol = solve_ivp(rhs, (0.0, 1.0), [a, b, *(vel / speed)], method="DOP853", rtol=1e-12, atol=1e-12)
        end = np.asarray(sol.y, dtype=np.float64)[:2, -1]
        out.append(case(f"{name}/geodesic", "geodesic", [uv[0], vel], end, rtol=1e-7, atol=1e-9, surface=name))
    for name, area, box in [
        ("sphere", 16 * np.pi, [np.pi, 2 * np.pi]),
        ("torus", 12 * np.pi**2, [2 * np.pi, 2 * np.pi]),
    ]:
        # Closed forms: 4 pi R^2 and 4 pi^2 R a.
        out.append(case(f"{name}/area", "area", [np.array(box)], np.array(area), rtol=1e-10, surface=name))
    return out
