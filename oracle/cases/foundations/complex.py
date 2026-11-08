"""Complex scalars and functions, Mobius maps, contour integrals and residues against numpy (complex arithmetic,
roots), scipy.integrate.quad and sympy (derivatives, residues, Laurent series)."""

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy
import sympy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.integrate import quad

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}

type ComplexArray = npt.NDArray[np.complex128]

z = sympy.Symbol("z")

FUNCTIONS: dict[str, sympy.Expr] = {
    "poles": 1 / (z * (z - 1)),
    "double": sympy.exp(z) / z**2,
    "cotangent": sympy.cos(z) / sympy.sin(z),
    "rational": (z**2 + 1) / (z**3 - 8),
}


def parts(w: ComplexArray) -> FloatArray:
    """Real and imaginary parts stacked along the first axis."""
    return np.stack([w.real, w.imag])


def value(e: sympy.Basic) -> complex:
    return complex(sympy.N(e, 20))


def cases() -> list[Case]:
    g = rng(1209)
    out: list[Case] = []
    w = g.normal(size=6) + 1j * g.normal(size=6)
    u = g.normal(size=6) + 1j * g.normal(size=6)
    inputs = [w.real, w.imag]
    out.append(case("log", "log", inputs, parts(np.log(w)), rtol=1e-13, atol=1e-15))
    out.append(case("sqrt", "sqrt", inputs, parts(np.sqrt(w)), rtol=1e-13, atol=1e-15))
    out.append(case("sin", "sin", inputs, parts(np.sin(w)), rtol=1e-13, atol=1e-15))
    out.append(case("cos", "cos", inputs, parts(np.cos(w)), rtol=1e-13, atol=1e-15))
    out.append(case("pow", "pow", [*inputs, u.real, u.imag], parts(np.power(w, u)), rtol=1e-12, atol=1e-14))
    out.append(case("polar", "polar", inputs, np.stack([np.abs(w), np.angle(w)]), rtol=1e-14))
    for i, (target, n) in enumerate([(-8.0 + 0j, 3), (1 + 1j, 5), (16.0 + 0j, 4)]):
        coeffs = np.zeros(n + 1, dtype=np.complex128)
        coeffs[0], coeffs[-1] = 1, -target
        roots: ComplexArray = np.roots(coeffs).astype(np.complex128)
        order = np.lexsort((roots.imag, roots.real))
        out.append(
            case(f"roots/{i}", "nthRoots", [np.array([target.real, target.imag, n])], parts(roots[order]), atol=1e-12)
        )

    grid_re = np.linspace(-2, 2, 7)
    grid_im = np.linspace(1.5, -1.5, 5)
    zz = grid_re[None, :] + 1j * grid_im[:, None]
    f = (zz**2 - 1) / (zz**2 + 1j)
    out.append(case("domain/modulus", "domainModulus", [], np.abs(f), rtol=1e-13))
    out.append(case("domain/argument", "domainArgument", [], np.angle(f), rtol=1e-13, atol=1e-14))

    a, b, c, d = 1 + 2j, -1j, 0.5 + 0j, 2 - 1j
    pts = g.normal(size=5) + 1j * g.normal(size=5)
    coef = np.array([a.real, a.imag, b.real, b.imag, c.real, c.imag, d.real, d.imag])
    out.append(case("mobius/apply", "mobiusApply", [coef, pts.real, pts.imag], parts((a * pts + b) / (c * pts + d))))
    fixed: ComplexArray = np.roots([c, d - a, -b]).astype(np.complex128)
    order = np.lexsort((fixed.imag, fixed.real))
    out.append(case("mobius/fixed", "mobiusFixed", [coef], parts(fixed[order]), rtol=1e-12, atol=1e-14))
    src = np.array([0, 1j, 2 + 0j])
    dst = np.array([1 + 1j, -1 + 0j, 3j])
    probe = np.array([0.5 + 0.5j, -1 + 2j])
    # The cross-ratio is preserved: (w, w1; w2, w3) = (z, z1; z2, z3), solved for w by sympy.
    wsym, zsym = sympy.symbols("w zz")

    def cross_ratio(p: sympy.Expr, q: complex, r: complex, s: complex) -> sympy.Expr:
        return (p - q) * (r - s) / ((p - s) * (r - q))

    images: list[complex] = []
    for pz in probe:
        sol = sympy.solve(sympy.Eq(cross_ratio(wsym, *dst), cross_ratio(zsym, *src).subs(zsym, complex(pz))), wsym)
        images.append(value(cast(list[sympy.Expr], sol)[0]))
    out.append(
        case(
            "mobius/points",
            "mobiusFromPoints",
            [parts(src), parts(dst), parts(probe)],
            parts(np.array(images)),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    for name, fz in [
        ("square", z**2),
        ("exp", sympy.exp(z)),
        ("joukowski", z + 1 / z),
        ("cayley", (z - sympy.I) / (z + sympy.I)),
    ]:
        point = 0.7 + 0.4j
        der = value(sympy.diff(fz, z).subs(z, point))
        out.append(
            case(
                f"derivative/{name}",
                "derivative",
                [np.array([point.real, point.imag])],
                np.array([der.real, der.imag]),
                rtol=1e-12,
                f=name,
            )
        )

    for name, fz in FUNCTIONS.items():
        poles = {"poles": 1.0 + 0j, "double": 0j, "cotangent": 0j, "rational": 2.0 + 0j}
        p = poles[name]
        res = value(sympy.residue(fz, z, sympy.nsimplify(p)))
        out.append(
            case(
                f"residue/{name}",
                "residue",
                [np.array([p.real, p.imag])],
                np.array([res.real, res.imag]),
                rtol=1e-9,
                atol=1e-11,
                f=name,
            )
        )
    exp_series = sympy.series(sympy.exp(z), z, 0, 6).removeO()
    laurent = [value(sympy.Poly(exp_series, z).coeff_monomial(z**k)).real for k in range(6)]
    out.append(case("laurent/exp", "laurentExp", [], np.array(laurent), rtol=1e-12, atol=1e-14))

    # Contour integral of z^2 along the parabola z(t) = t + i t^2, t in [0, 1]: (1 + i)^3 / 3.
    def piece(t: float, part: int) -> float:
        zt = t + 1j * t**2
        val = zt**2 * (1 + 2j * t)
        return float(val.real if part == 0 else val.imag)

    re = quad(piece, 0.0, 1.0, args=(0,), epsabs=1e-13)[0]
    im = quad(piece, 0.0, 1.0, args=(1,), epsabs=1e-13)[0]
    out.append(case("contour/parabola", "contourParabola", [], np.array([re, im]), rtol=1e-11))

    poly = np.array([1, -2 + 1j, 0.5, 1j, -0.3])
    found: ComplexArray = np.roots(poly).astype(np.complex128)
    for radius in [0.5, 1.0, 1.5, 3.0]:
        inside = int(np.sum(np.abs(found) < radius))
        out.append(
            case(f"argument/{radius}", "argumentPrinciple", [parts(poly), np.array(radius)], np.array(float(inside)))
        )
    return out
