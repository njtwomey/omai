"""Equations against SymPy's `solve`: the solutions as sorted numeric values (real first), and linear systems as exact
expressions per unknown."""

import sympy as sp
from mdml_oracle.fixture import Case, Json, case
from mdml_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

x = SYMBOLS["x"]

EQUATIONS: list[str] = [
    "x**2 - 5*x + 6",
    "2*x + 3",
    "x**2 + 1",
    "x**3 - 6*x**2 + 11*x - 6",
    "x**4 - 5*x**2 + 4",
    "x**3 - 2",
    "(x - 1)/(x + 2)",
    "(x**2 - 1)/(x - 1)",
    "exp(x) - 2",
    "exp(2*x) - 3*exp(x) + 2",
    "log(x) - 1",
    "2**x - 8",
    "sin(x)",
    "cos(x)",
    "sin(x) - 1/2",
    "tan(x) - 1",
    "x*exp(x) - 0",
    "sqrt(x) - 3",
    "x**2 - 2*x - 1",
    "1/x - 2",
    "exp(x)*(x - 1)",
    "x**6 - 1",
]

SYSTEMS: list[tuple[list[str], list[str]]] = [
    (["x + y - 3", "x - y - 1"], ["x", "y"]),
    (["2*x + 3*y - z - 1", "x - y + 2*z - 3", "3*x + y + z - 4"], ["x", "y", "z"]),
    (["a*x + y - 1", "x - y"], ["x", "y"]),
    (["x + y + z - 1", "x - y"], ["x", "y"]),
]


def numeric(sols: list[sp.Expr]) -> list[Json]:
    """Solutions as (re, im) pairs, real ones first in increasing order."""
    vals = [complex(s) for s in sols]
    keyed = sorted(vals, key=lambda z: (abs(z.imag) > 1e-12, z.real, z.imag))
    return [[z.real, z.imag] for z in keyed]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, s in enumerate(EQUATIONS):
        sols = sp.solve(parse(s), x)
        out.append(case(f"solve/{i}", "solve", [s], numeric(sols), rtol=1e-9, atol=1e-9, text=[str(v) for v in sols]))
    for i, (eqs, names) in enumerate(SYSTEMS):
        sol = sp.solve([parse(e) for e in eqs], [SYMBOLS[n] for n in names], dict=True)[0]
        for name in names:
            if SYMBOLS[name] in sol:
                out.append(
                    expr_case(
                        f"linear/{i}/{name}",
                        "solveLinearSystem",
                        [";".join(eqs), ",".join(names), name],
                        sol[SYMBOLS[name]],
                    )
                )
    return out
