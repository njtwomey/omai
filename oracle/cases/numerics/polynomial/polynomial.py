"""Polynomials against numpy.polynomial, scipy.special orthogonal polynomials and scipy.signal partial fractions."""

from collections.abc import Callable

import numpy as np
import numpy.polynomial.chebyshev as C
import numpy.polynomial.hermite as H
import numpy.polynomial.hermite_e as HE
import numpy.polynomial.laguerre as L
import numpy.polynomial.legendre as LG
import numpy.polynomial.polynomial as P
import numpy.typing as npt
import scipy
import scipy.signal as sg
import scipy.special as sp
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

# numpy's family functions have overloaded signatures; results are converted with np.asarray where used.
type Any_ = Callable[..., object]

FAMILIES: dict[str, tuple[Any_, Any_, Any_, Any_]] = {
    "legendre": (LG.legval, LG.legfit, LG.leg2poly, LG.poly2leg),
    "chebyshev": (C.chebval, C.chebfit, C.cheb2poly, C.poly2cheb),
    "hermite": (H.hermval, H.hermfit, H.herm2poly, H.poly2herm),
    "hermite_e": (HE.hermeval, HE.hermefit, HE.herme2poly, HE.poly2herme),
    "laguerre": (L.lagval, L.lagfit, L.lag2poly, L.poly2lag),
}


def canonical(r: npt.ArrayLike, p: npt.ArrayLike) -> FloatArray:
    """(pole re, pole im, power, residue re, residue im) rows sorted by pole then power."""
    r = np.asarray(r, dtype=np.complex128)
    p = np.asarray(p, dtype=np.complex128)
    rows: list[list[float]] = []
    power = 0
    for i in range(len(p)):
        power = power + 1 if i > 0 and p[i] == p[i - 1] else 1
        rows.append(
            [float(np.real(p[i])), float(np.imag(p[i])), float(power), float(np.real(r[i])), float(np.imag(r[i]))]
        )
    out = np.array(rows, dtype=np.float64)
    return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]


def cases() -> list[Case]:
    g = rng(80)
    out: list[Case] = []
    c = g.normal(size=6)
    x = np.linspace(-1.5, 1.5, 13)
    out.append(case("polyval", "polyval", [x, c], P.polyval(x, c), rtol=1e-12, atol=1e-13))
    xs = np.sort(g.uniform(-2, 2, size=30))
    ys = np.sin(xs) + 0.1 * g.normal(size=30)
    w = g.uniform(0.5, 2, size=30)
    out.append(case("polyfit", "polyfit", [xs, ys], P.polyfit(xs, ys, 4), rtol=1e-9, atol=1e-10, deg=4))
    out.append(
        case("polyfit/weighted", "polyfit", [xs, ys, w], P.polyfit(xs, ys, 4, w=w), rtol=1e-9, atol=1e-10, deg=4)
    )
    out.append(case("polyder", "polyder", [c], P.polyder(c, 2, scl=0.5), rtol=1e-13, m=2, scl=0.5))
    out.append(case("polyint", "polyint", [c], P.polyint(c, 2, k=[1.0, -2.0], lbnd=0.5), rtol=1e-12, atol=1e-13))
    d = g.normal(size=3)
    out.append(case("polymul", "polymul", [c, d], P.polymul(c, d), rtol=1e-13))
    q, r = P.polydiv(c, d)
    out.append(case("polydiv", "polydiv", [c, d], np.concatenate([q, r]), rtol=1e-10, atol=1e-12))
    roots = P.polyroots(c)
    out.append(case("polyroots", "polyroots", [c], np.stack([roots.real, roots.imag]), rtol=1e-9, atol=1e-10))
    out.append(case("polyfromroots", "polyfromroots", [xs[:5]], P.polyfromroots(xs[:5]), rtol=1e-12, atol=1e-13))

    for name, (val, fit, to_poly, from_poly) in FAMILIES.items():
        xq = np.linspace(0, 4, 11) if name == "laguerre" else x
        out.append(
            case(
                f"val/{name}",
                "orthoval",
                [xq, c],
                np.asarray(val(xq, c), dtype=np.float64),
                rtol=1e-11,
                atol=1e-11,
                family=name,
            )
        )
        xf = np.linspace(0, 6, 30) if name == "laguerre" else xs
        out.append(
            case(
                f"fit/{name}",
                "orthofit",
                [xf, ys],
                np.asarray(fit(xf, ys, 5), dtype=np.float64),
                rtol=1e-8,
                atol=1e-9,
                family=name,
            )
        )
        out.append(
            case(
                f"toPower/{name}",
                "orthoToPower",
                [c],
                np.asarray(to_poly(c), dtype=np.float64),
                rtol=1e-11,
                atol=1e-11,
                family=name,
            )
        )
        out.append(
            case(
                f"fromPower/{name}",
                "powerToOrtho",
                [c],
                np.asarray(from_poly(c), dtype=np.float64),
                rtol=1e-11,
                atol=1e-11,
                family=name,
            )
        )
    for n, alpha, beta in [(3, 0.5, -0.3), (6, 1.5, 2.0)]:
        xj = np.linspace(-1, 1, 9)
        ref = sum(c[k] * sp.eval_jacobi(k, alpha, beta, xj) for k in range(n))
        out.append(
            case(
                f"val/jacobi/{n}",
                "orthoval",
                [xj, c[:n]],
                np.asarray(ref, dtype=np.float64),
                rtol=1e-11,
                atol=1e-12,
                family="jacobi",
                alpha=alpha,
                beta=beta,
            )
        )

    systems = {
        "simple": (np.array([1.0, 2.0]), np.array([1.0, 6.0, 11.0, 6.0])),
        "repeated": (np.array([2.0, -1.0, 3.0]), np.poly([-1.0, -1.0, -1.0, 2.0])),
        "complex": (np.array([1.0, 0.0, 1.0]), np.array([1.0, 2.0, 5.0, 0.0])),
        "improper": (np.array([3.0, 1.0, 4.0, 1.0, 5.0]), np.array([1.0, -1.0, -2.0])),
    }
    for name, (b, a) in systems.items():
        r, p, k = sg.residue(b, a)
        out.append(case(f"residue/{name}", "residue", [b, a], canonical(r, p), rtol=1e-8, atol=1e-9))
        out.append(
            case(
                f"residue/{name}/direct",
                "residueDirect",
                [b, a],
                np.atleast_1d(np.asarray(k, dtype=np.float64)),
                rtol=1e-12,
                atol=1e-12,
            )
        )
        nb, na = sg.invres(np.asarray(r, dtype=np.complex128), np.asarray(p, dtype=np.complex128), k)
        out.append(case(f"invres/{name}", "invres", [b, a], np.real(np.concatenate([nb, na])), rtol=1e-8, atol=1e-9))
    zsys = {
        "simple": (np.array([1.0, 0.5]), np.array([1.0, -0.9, 0.2])),
        "repeated": (np.array([1.0]), np.array([1.0, -1.0, 0.25])),
        "improper": (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, -0.3])),
    }
    for name, (b, a) in zsys.items():
        r, p, k = sg.residuez(b, a)
        out.append(case(f"residuez/{name}", "residuez", [b, a], canonical(r, p), rtol=1e-8, atol=1e-9))
        out.append(
            case(
                f"residuez/{name}/direct",
                "residuezDirect",
                [b, a],
                np.atleast_1d(np.asarray(k, dtype=np.float64)),
                rtol=1e-12,
                atol=1e-12,
            )
        )
    return out
