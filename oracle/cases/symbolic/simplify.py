"""Rewrites against SymPy: expand, factor (over the rationals, several variables), collect, together, cancel, apart,
trigsimp and simplify. Each output carries SymPy's text and values; `exact` marks the cases whose printed form
mdml-js reproduces, the others (trigsimp and simplify, whose choices between equivalent forms differ) are compared
by value."""

import sympy as sp
from mdml_oracle.fixture import Case
from mdml_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

x = SYMBOLS["x"]

EXPAND: list[str] = [
    "(x + y)**2",
    "(x + y + 1)**3",
    "(x - 1)*(x + 1)*(x**2 + 1)",
    "(2*x + 3)**4",
    "x*(x + 1)**2 - x**3",
    "(x + 1)/(x + 2)",
    "(x + 1)**(-2)",
    "exp(x + y)",
    "x**(y + 2)",
    "(a + b)*(c + x) + (a - b)**2",
    "sin(x)*(cos(x) + 1)",
    "(x + sqrt(2))**2",
]

FACTOR: list[str] = [
    "x**2 - 1",
    "2*x**2 - 2",
    "x**4 - 1",
    "x**6 - 1",
    "x**4 + 4",
    "x**2 - 2",
    "x**3 - x**2*y - x*y**2 + y**3",
    "x**2 - y**2",
    "1 - x",
    "x**8 - 1",
    "(x**2 + 2*x + 1)/(x**2 - 1)",
    "x**3 + 3*x**2 + 3*x + 1",
    "6*x**2 + 5*x + 1",
    "x**5 - x**4 - x + 1",
    "x**2*y + x*y**2 + x + y",
    "a**2*b - a*b**2",
    "x**4 + x**2 + 1",
    "x**2/2 - 1/2",
    "sin(x)**2 - 1",
    "x**3 - 6*x**2 + 11*x - 6",
    "x**10 - 1",
    "x**4 - 5*x**2 + 6",
    "x*y + x + y + 1",
    "x**2 + 2*x*y + y**2 - z**2",
]

COLLECT: list[tuple[str, str]] = [
    ("a*x**2 + b*x**2 + a*x - b*x + c", "x"),
    ("x*y + x - 3 + 2*x**2 - z*x**2 + x**3", "x"),
    ("a*x + b*x + a*y + b*y", "x"),
]

TOGETHER: list[str] = [
    "1/x + 1/y",
    "1/(x + 1) + 1/(x + 1)**2",
    "x/2 + 1/3",
    "1/x + 1/x**2 + 1",
    "a/b + c/d",
    "1/(x + 1) + x/(x + 1)",
]

CANCEL: list[str] = [
    "(x**2 - 1)/(x - 1)",
    "(2*x + 2)/(4*x)",
    "(x**3 - 1)/(x**2 - 1)",
    "(x**2 + 2*x*y + y**2)/(x + y)",
    "(x*y - x)/(y**2 - 1)",
    "1/x + 1/(x*(x + 1))",
]

APART: list[str] = [
    "1/(x**2 - 1)",
    "(x + 2)/(x**2 + x)",
    "1/(x*(x + 1)**2)",
    "(x**3 + 1)/(x**2 - 4)",
    "1/(x**2 + 1)/(x - 1)",
    "(2*x + 3)/(x**2 + 2*x + 5)",
    "1/(x**4 - 1)",
    "x**2/(x - 1)**3",
    "1/((x - a)*(x - b))",
]

TRIGSIMP: list[str] = [
    "sin(x)**2 + cos(x)**2",
    "2*sin(x)*cos(x)",
    "sin(x)/cos(x)",
    "1 - sin(x)**2",
    "cos(x)**2 - sin(x)**2",
    "tan(x)*cos(x)",
    "sin(x)**4 - cos(x)**4",
    "cosh(x)**2 - sinh(x)**2",
    "sin(x)*cos(y) + cos(x)*sin(y)",
    "1/cos(x)**2 - 1",
]

SIMPLIFY: list[str] = [
    "(x**2 - 1)/(x - 1)",
    "sin(x)**2 + cos(x)**2 + x",
    "(x**3 + x**2 - x - 1)/(x**2 + 2*x + 1)",
    "exp(x)*exp(y)",
    "(x + 1)**2 - (x - 1)**2",
    "1/x + 1/(x*(x + 1))",
    "x**2*x**3",
    "2*sin(x)*cos(x) + 1",
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, s in enumerate(EXPAND):
        out.append(expr_case(f"expand/{i}", "expand", [s], sp.expand(parse(s))))
    for i, s in enumerate(FACTOR):
        out.append(expr_case(f"factor/{i}", "factor", [s], sp.factor(parse(s))))
    for i, (s, v) in enumerate(COLLECT):
        out.append(expr_case(f"collect/{i}", "collect", [s, v], sp.collect(parse(s), parse(v))))
    for i, s in enumerate(TOGETHER):
        out.append(expr_case(f"together/{i}", "together", [s], sp.together(parse(s))))
    for i, s in enumerate(CANCEL):
        out.append(expr_case(f"cancel/{i}", "cancel", [s], sp.cancel(parse(s))))
    for i, s in enumerate(APART):
        out.append(expr_case(f"apart/{i}", "apart", [s, "x"], sp.apart(parse(s), x)))
    for i, s in enumerate(TRIGSIMP):
        out.append(expr_case(f"trigsimp/{i}", "trigsimp", [s], sp.trigsimp(parse(s)), exact=False))
    for i, s in enumerate(SIMPLIFY):
        out.append(expr_case(f"simplify/{i}", "simplify", [s], sp.simplify(parse(s)), exact=False))
    return out
