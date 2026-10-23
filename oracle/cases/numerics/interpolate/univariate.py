"""One-dimensional interpolation against scipy.interpolate."""

from typing import Literal

import numpy as np
import scipy
import scipy.interpolate as si
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Condition = tuple[Literal[1, 2], FloatArray]

BOUNDARIES: dict[str, Literal["not-a-knot", "natural", "clamped", "periodic"] | tuple[Condition, Condition]] = {
    "not-a-knot": "not-a-knot",
    "natural": "natural",
    "clamped": "clamped",
    "periodic": "periodic",
    "explicit": ((1, np.array(0.5)), (2, np.array(-1.0))),
}


def data(seed: int, n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    g = rng(seed)
    x = np.sort(g.uniform(0, 5, size=n))
    x[0], x[-1] = 0.0, 5.0
    y = np.sin(x) + 0.3 * g.normal(size=n)
    q = np.linspace(-0.5, 5.5, 23)
    return x, y, q


def cases() -> list[Case]:
    out: list[Case] = []
    x, y, q = data(50, 9)
    inside = np.linspace(0, 5, 17)
    for name, bc in BOUNDARIES.items():
        yy = y.copy()
        if name == "periodic":
            yy[-1] = yy[0]
        s = si.CubicSpline(x, yy, bc_type=bc)
        for nu in (0, 1, 2, 3):
            out.append(
                case(
                    f"cubicSpline/{name}/nu{nu}",
                    "cubicSpline",
                    [x, yy, q],
                    s(q, nu),
                    rtol=1e-10,
                    atol=1e-10,
                    bc=name,
                    nu=nu,
                )
            )
        out.append(
            case(
                f"cubicSpline/{name}/integrate",
                "cubicSplineIntegrate",
                [x, yy],
                np.array(s.integrate(0.3, 4.1)),
                rtol=1e-10,
                bc=name,
            )
        )
        out.append(
            case(
                f"cubicSpline/{name}/antiderivative",
                "cubicSplineAnti",
                [x, yy, q],
                s.antiderivative()(q),
                rtol=1e-10,
                atol=1e-10,
                bc=name,
            )
        )
    s = si.CubicSpline(x, y)
    out.append(
        case(
            "cubicSpline/roots", "cubicSplineRoots", [x, y], np.sort(s.roots(extrapolate=False)), rtol=1e-9, atol=1e-10
        )
    )
    for n in (2, 3, 4):
        xs, ys, qs = data(51 + n, n)
        out.append(
            case(
                f"cubicSpline/small{n}",
                "cubicSpline",
                [xs, ys, qs],
                si.CubicSpline(xs, ys)(qs),
                rtol=1e-10,
                atol=1e-10,
                bc="not-a-knot",
                nu=0,
            )
        )
    y2 = np.stack([y, np.cos(x)], axis=1)
    out.append(
        case(
            "cubicSpline/2d",
            "cubicSpline",
            [x, y2, q],
            si.CubicSpline(x, y2)(q),
            rtol=1e-10,
            atol=1e-10,
            bc="not-a-knot",
            nu=0,
        )
    )

    dydx = np.cos(x)
    out.append(
        case(
            "cubicHermite",
            "cubicHermite",
            [x, y, dydx, q],
            si.CubicHermiteSpline(x, y, dydx)(q),
            rtol=1e-10,
            atol=1e-10,
        )
    )
    out.append(case("pchip", "pchip", [x, y, q], si.PchipInterpolator(x, y)(q), rtol=1e-10, atol=1e-10))
    steps = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 2.0])
    out.append(case("pchip/steps", "pchip", [x, steps, q], si.PchipInterpolator(x, steps)(q), rtol=1e-10, atol=1e-10))
    for method in ("akima", "makima"):
        for name, yy in (("data", y), ("steps", steps)):
            a = si.Akima1DInterpolator(x, yy, method=method)
            out.append(case(f"{method}/{name}", method, [x, yy, inside], a(inside), rtol=1e-10, atol=1e-10))

    t = np.array([0, 0, 0, 0, 1.2, 2.0, 3.5, 5, 5, 5, 5], dtype=np.float64)
    c = rng(52).normal(size=7)
    b = si.BSpline(t, c, 3)
    for nu in (0, 1, 2):
        out.append(case(f"bspline/nu{nu}", "bspline", [t, c, q], b(q, nu), rtol=1e-10, atol=1e-10, nu=nu))
    out.append(case("bspline/integrate", "bsplineIntegrate", [t, c], np.array(b.integrate(0.4, 4.4)), rtol=1e-10))
    out.append(
        case(
            "bspline/design",
            "bsplineDesign",
            [t, inside],
            si.BSpline.design_matrix(inside, t, 3).toarray(),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    for k in (1, 3, 5):
        out.append(
            case(
                f"makeInterpSpline/k{k}",
                "makeInterpSpline",
                [x, y, q],
                si.make_interp_spline(x, y, k=k)(q),
                rtol=1e-9,
                atol=1e-9,
                k=k,
            )
        )
    xm = np.sort(rng(53).uniform(0, 5, size=40))
    ym = np.sin(xm) + 0.1 * rng(54).normal(size=40)
    tk = np.r_[[0.0] * 4, [1.0, 2.0, 3.0, 4.0], [5.0] * 4]
    xm[0], xm[-1] = 0.0, 5.0
    wm = rng(55).uniform(0.5, 2, size=40)
    out.append(
        case(
            "makeLsqSpline",
            "makeLsqSpline",
            [xm, ym, tk, wm, inside],
            si.make_lsq_spline(xm, ym, tk, k=3, w=wm)(inside),
            rtol=1e-10,
        )
    )
    for lam in (1e-3, 0.1, 10.0):
        sp = si.make_smoothing_spline(xm, ym, lam=lam)
        out.append(
            case(f"smoothing/lam{lam}", "smoothing", [xm, ym, inside], sp(inside), rtol=1e-8, atol=1e-9, lam=lam)
        )
    out.append(
        case(
            "smoothing/gcv",
            "smoothingGcv",
            [xm, ym, inside],
            si.make_smoothing_spline(xm, ym)(inside),
            rtol=1e-3,
            atol=1e-3,
        )
    )
    out.append(
        case(
            "univariate/s0",
            "univariateS0",
            [x, y, inside],
            si.UnivariateSpline(x, y, s=0)(inside),
            rtol=1e-9,
            atol=1e-9,
        )
    )

    nodes = np.array([-1.0, -0.4, 0.1, 0.5, 1.2, 2.0])
    vals = np.exp(nodes) * np.cos(2 * nodes)
    pts = np.linspace(-1.2, 2.2, 15)
    out.append(
        case(
            "barycentric",
            "barycentric",
            [nodes, vals, pts],
            si.BarycentricInterpolator(nodes, vals)(pts),
            rtol=1e-11,
            atol=1e-12,
        )
    )
    out.append(
        case("lagrange", "lagrange", [nodes, vals], np.polyfit(nodes, vals, len(nodes) - 1), rtol=1e-9, atol=1e-9)
    )
    return out
