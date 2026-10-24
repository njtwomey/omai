"""Limits against SymPy's `limit`: removable singularities, l'Hopital forms, limits at infinity, exponential and
logarithmic forms, and one-sided limits at poles. The value is SymPy's exact result (text and value)."""

import sympy as sp
from whitebox_oracle.fixture import Case
from whitebox_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

x = SYMBOLS["x"]

LIMITS: list[tuple[str, str, str]] = [
    ("sin(x)/x", "0", "+"),
    ("(1 - cos(x))/x**2", "0", "+"),
    ("(exp(x) - 1)/x", "0", "+"),
    ("(1 + 1/x)**x", "oo", "-"),
    ("x*log(x)", "0", "+"),
    ("x**x", "0", "+"),
    ("1/x", "0", "+"),
    ("1/x", "0", "-"),
    ("(x**2 - 1)/(x - 1)", "1", "+"),
    ("(2*x**2 + 3)/(x**2 - x)", "oo", "-"),
    ("x*exp(-x)", "oo", "-"),
    ("log(x)/x", "oo", "-"),
    ("(sqrt(x + 1) - 1)/x", "0", "+"),
    ("tan(x)/x", "0", "+"),
    ("(x - sin(x))/x**3", "0", "+"),
    ("exp(-1/x)", "0", "+"),
    ("atan(x)", "oo", "-"),
    ("sqrt(x**2 + x) - x", "oo", "-"),
    ("x**2*exp(x)", "-oo", "+"),
    ("log(1 + x)/x", "0", "+"),
    ("(1 + 2*x)**(1/x)", "0", "+"),
    ("sin(3*x)/sin(2*x)", "0", "+"),
    ("1/x**2", "0", "+"),
    ("x/abs(x)", "0", "-"),
    ("cos(x)", "pi", "+"),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (s, point, d) in enumerate(LIMITS):
        value = sp.limit(parse(s), x, parse(point), d)
        out.append(expr_case(f"limit/{i}", "limit", [s, point, d], value))
    return out
