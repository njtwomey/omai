"""Text parsing against SymPy's `parse_expr` with the standard transformations plus `convert_xor` (^ is a power)
and `implicit_multiplication` (2x, 3(x + 1), (x + 1)(x - 1)). The LaTeX parser has no SymPy counterpart here
(`parse_latex` needs the ANTLR runtime), so it is checked by unit tests and round trips through `toLatex`."""

import sympy as sp
from mdml_oracle.fixture import Case
from mdml_oracle.symbolic import SYMBOLS, expr_case
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

TRANSFORMS = (*standard_transformations, convert_xor, implicit_multiplication)

INPUTS: list[str] = [
    "x^2 + 2x + 1",
    "2x^2 - 3x",
    "3(x + 1)",
    "(x + 1)(x - 1)",
    "2 x y",
    "x^-2",
    "-x^2",
    "2^3^2",
    "x/2/y",
    "x - y - z",
    "sqrt(x)^3",
    "exp(-x^2/2)",
    "sin(x)^2 + cos(x)^2",
    "log(x, 2)",
    "Abs(x - 1)",
    "E^x",
    "pi*x^2",
    "I*x + 1",
    "1/(1 + x^2)",
    "x**(1/3)",
    "factorial(5)",
    "3!",
    "x! + 1",
    "atan2(y, x)",
    "Eq(x^2, 4)",
    "x >= 1",
]


def parse_text(s: str) -> sp.Basic:
    """SymPy's parse of the text with the transformations above."""
    return parse_expr(s, local_dict=dict(SYMBOLS), transformations=TRANSFORMS)


def cases() -> list[Case]:
    return [expr_case(f"parse/{i}", "parse", [s], parse_text(s)) for i, s in enumerate(INPUTS)]
