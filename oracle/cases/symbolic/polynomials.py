"""Polynomials against SymPy: division (quotient and remainder), gcd and lcm, degrees and coefficients, Sturm
sequences, real root counts, resultants and discriminants (text), and roots (numeric values, sorted real first)."""

import numpy as np
import sympy as sp
from whitebox_oracle.fixture import Case, Json, case
from whitebox_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__, "numpy": np.__version__}

x = SYMBOLS["x"]

DIV: list[tuple[str, str]] = [
    ("x**3 + 2*x + 1", "x - 1"),
    ("x**4 - 1", "x**2 + 1"),
    ("2*x**3 + 3*x**2 - 1", "2*x + 1"),
    ("x**5", "x**2 + x + 1"),
]

GCD: list[tuple[str, str]] = [
    ("x**2 - 1", "x**2 - 3*x + 2"),
    ("x**2*y - y", "x*y**2 + y**2"),
    ("6*x**2 + 12*x + 6", "4*x + 4"),
    ("x**4 - y**4", "x**2 + 2*x*y + y**2"),
    ("x**3 - 1", "x**2 + x + 1"),
]

ROOTS: list[str] = [
    "x**2 - 3*x + 2",
    "x**3 - x**2 - 2*x + 2",
    "x**3 - 3*x + 1",
    "x**3 + x + 1",
    "x**4 + x + 1",
    "x**4 - 10*x**2 + 1",
    "x**4 + 2*x**3 - x + 3",
    "x**5 - 2",
    "(x - 1)**3*(x + 2)",
    "x**2 + x + 1",
    "x**6 - 1",
    "2*x**4 - 3*x**3 + x - 7",
]

STURM: list[str] = ["x**3 - 2*x + 1", "x**4 - 3*x**2 + 1", "x**2 + 1", "x**5 - x + 1"]

COUNT: list[tuple[str, float, float]] = [
    ("x**3 - 2*x + 1", -10.0, 10.0),
    ("x**2 + 1", -10.0, 10.0),
    ("(x - 1)**2*(x + 2)", 0.0, 1.0),
    ("x**5 - x + 1", -2.0, 2.0),
    ("x**4 - 10*x**2 + 1", 0.0, 4.0),
]

RESULTANT: list[tuple[str, str]] = [
    ("x**2 + y", "x - y"),
    ("x**2 - 2", "x**3 - 3*x"),
    ("a*x**2 + b*x + c", "2*a*x + b"),
    ("x**2 + y**2 - 1", "x - y"),
]

DISCRIMINANT: list[str] = ["a*x**2 + b*x + c", "x**3 + p*x + q", "x**4 + 1", "x**3 - 3*x + 1"]


def root_key(z: complex) -> tuple[bool, float, float]:
    """Real roots first, then by real part (rounded so that conjugates tie) and imaginary part."""
    return (abs(z.imag) > 1e-9, round(z.real, 9), z.imag)


def sorted_roots(f: sp.Expr) -> list[Json]:
    """Roots as (re, im) pairs with multiplicity, real roots first in increasing order: numpy's roots of each
    irreducible factor over the rationals (so repeated roots are exact rather than numerically split)."""
    rs: list[complex] = []
    for g, k in sp.factor_list(f)[1]:
        cs = np.asarray([complex(c) for c in sp.Poly(g, x).all_coeffs()], dtype=np.complex128)
        rs.extend([complex(z) for z in np.roots(cs)] * int(k))
    keyed = sorted(rs, key=root_key)
    return [[z.real, 0.0 if abs(z.imag) < 1e-12 else z.imag] for z in keyed]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (f, g) in enumerate(DIV):
        q, r = sp.div(parse(f), parse(g), x)
        out.append(expr_case(f"div/{i}/quotient", "polyDivQuotient", [f, g], q))
        out.append(expr_case(f"div/{i}/remainder", "polyDivRemainder", [f, g], r))
    for i, (f, g) in enumerate(GCD):
        out.append(expr_case(f"gcd/{i}", "polyGcd", [f, g], sp.gcd(parse(f), parse(g))))
        out.append(expr_case(f"lcm/{i}", "polyLcm", [f, g], sp.lcm(parse(f), parse(g))))
    for i, f in enumerate(ROOTS):
        e = parse(f)
        rts = sp.roots(sp.Poly(e, x))
        mult = int(sum(rts.values()))
        out.append(case(f"roots/{i}", "exactRoots", [f], sorted_roots(e), rtol=1e-8, atol=1e-8, found=mult))
    for i, f in enumerate(STURM):
        seq = [p.as_expr() for p in sp.sturm(sp.Poly(parse(f), x))]
        out.append(case(f"sturm/{i}", "sturm", [f], [str(s) for s in seq]))
    for i, (f, a, b) in enumerate(COUNT):
        n = sp.Poly(parse(f), x).count_roots(a, b)
        out.append(case(f"countRoots/{i}", "countRealRoots", [f, a, b], int(n)))
    for i, (f, g) in enumerate(RESULTANT):
        out.append(expr_case(f"resultant/{i}", "resultant", [f, g], sp.expand(sp.resultant(parse(f), parse(g), x))))
    for i, f in enumerate(DISCRIMINANT):
        out.append(expr_case(f"discriminant/{i}", "discriminant", [f], sp.expand(sp.discriminant(parse(f), x))))
    return out
