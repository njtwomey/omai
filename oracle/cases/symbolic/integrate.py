"""Integrals against SymPy's `integrate`. Antiderivatives are compared by value up to a constant (SymPy and
whitebox-js may choose different but equivalent forms, such as log(sec + tan) against a difference of logs), and every
whitebox-js antiderivative is also differentiated back to the integrand in the test. Definite integrals compare the
exact value."""

import sympy as sp
from whitebox_oracle.fixture import Case
from whitebox_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

x = SYMBOLS["x"]

INDEFINITE: list[str] = [
    "x**3",
    "1/x",
    "exp(2*x + 1)",
    "sin(3*x)",
    "cos(x)**2",
    "sin(x)**3",
    "sin(x)**2*cos(x)**3",
    "tan(x)",
    "sec(x)**2",
    "x*exp(x)",
    "x**2*exp(-x)",
    "x*cos(x)",
    "x**2*sin(x)",
    "log(x)",
    "x*log(x)",
    "atan(x)",
    "asin(x)",
    "exp(x)*sin(x)",
    "exp(2*x)*cos(3*x)",
    "1/(x**2 + 1)",
    "1/(x**2 - 1)",
    "(x + 2)/(x**2 + x)",
    "1/(x*(x + 1)**2)",
    "(2*x + 3)/(x**2 + 2*x + 5)",
    "1/(x**2 + 1)**2",
    "x/(x**2 + 1)",
    "x*exp(x**2)",
    "cos(x)*exp(sin(x))",
    "log(x)/x",
    "1/sqrt(1 - x**2)",
    "1/sqrt(x**2 + 1)",
    "sqrt(1 - x**2)",
    "exp(-x**2)",
    "tan(x)**3",
    "sin(x)*cos(2*x)",
    "x*sinh(x)",
    "(x**3 + 1)/(x - 1)",
    "x*sqrt(x + 1)",
    "a*x**2 + b*x + c",
    "1/(a*x + b)",
]

DEFINITE: list[tuple[str, str, str]] = [
    ("x**2", "0", "3"),
    ("exp(-x)", "0", "oo"),
    ("sin(x)", "0", "pi"),
    ("1/(x**2 + 1)", "-oo", "oo"),
    ("x*exp(-x**2)", "0", "oo"),
    ("log(x)", "1", "E"),
    ("exp(-x**2)", "-oo", "oo"),
    ("1/x", "1", "2"),
    ("cos(x)**2", "0", "pi"),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, s in enumerate(INDEFINITE):
        out.append(expr_case(f"integrate/{i}", "integrate", [s], sp.integrate(parse(s), x), exact=False))
    for i, (s, a, b) in enumerate(DEFINITE):
        value = sp.integrate(parse(s), (x, parse(a), parse(b)))
        out.append(expr_case(f"definite/{i}", "definiteIntegral", [s, a, b], value))
    return out
