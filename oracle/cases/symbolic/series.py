"""Series against SymPy: `series` (Taylor and Laurent at 0, other points and infinity; printed form and values of the
truncated series), residues, and Pade approximants from SymPy's Taylor coefficients. scipy.interpolate.pade is
deprecated (its suggested replacement is mpmath), so the Pade reference solves the same linear system with
numpy.linalg.solve."""

from typing import Any, cast

import numpy as np
import sympy as sp
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray
from mdml_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__, "numpy": np.__version__}

x = SYMBOLS["x"]

SERIES: list[tuple[str, str, int]] = [
    ("sin(x)", "0", 6),
    ("cos(x)", "0", 8),
    ("exp(x)", "0", 5),
    ("log(1 + x)", "0", 5),
    ("tan(x)", "0", 8),
    ("1/(1 - x)", "0", 5),
    ("sqrt(1 + x)", "0", 4),
    ("exp(sin(x))", "0", 5),
    ("1/sin(x)", "0", 4),
    ("1/(x*(1 - x))", "0", 3),
    ("cot(x)", "0", 4),
    ("atan(x)", "0", 8),
    ("exp(x)", "1", 3),
    ("log(x)", "1", 4),
    ("sin(x)", "pi", 4),
    ("1/(x + 1)", "oo", 4),
    ("x/(x**2 + 1)", "oo", 6),
    ("exp(x)/x**2", "0", 2),
    ("asin(x)", "0", 6),
    ("cosh(x)", "0", 6),
    ("(1 + x)**(1/3)", "0", 4),
    ("exp(a*x)", "0", 4),
]

RESIDUE: list[str] = ["exp(x)/x**2", "1/sin(x)", "cos(x)/x**3", "1/(x*(x - 1))"]

PADE: list[tuple[str, int, int]] = [("exp(x)", 2, 2), ("log(1 + x)", 2, 1), ("cos(x)", 2, 2), ("atan(x)", 3, 2)]


def pade_coefficients(c: FloatArray, m: int, n: int) -> tuple[FloatArray, FloatArray]:
    """[m/n] Pade coefficients (lowest degree first, q[0] = 1) from Taylor coefficients c[0..m+n]:
    sum_j q_j c_(k-j) = 0 for k = m+1..m+n, then p_k = sum_j q_j c_(k-j)."""
    a = np.array([[c[k - j] if k - j >= 0 else 0.0 for j in range(1, n + 1)] for k in range(m + 1, m + n + 1)])
    rhs = -np.array([c[k] for k in range(m + 1, m + n + 1)])
    q = np.concatenate([[1.0], np.linalg.solve(a, rhs)]) if n > 0 else np.array([1.0])
    p = np.array([sum(q[j] * c[k - j] for j in range(min(k, n) + 1)) for k in range(m + 1)])
    return p, q


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (s, point, n) in enumerate(SERIES):
        res = sp.series(parse(s), x, cast(Any, parse(point)), n)
        out.append(expr_case(f"series/{i}", "series", [s, point, str(n)], res.removeO(), order=str(res.getO())))
    for i, s in enumerate(RESIDUE):
        out.append(expr_case(f"residue/{i}", "residue", [s], sp.residue(parse(s), x, 0)))
    for i, (s, m, n) in enumerate(PADE):
        taylor = sp.series(parse(s), x, 0, m + n + 1).removeO()
        coeffs = np.array([float(taylor.coeff(x, k)) for k in range(m + n + 1)], dtype=np.float64)
        p, q = pade_coefficients(coeffs, m, n)
        # Highest power first, as numpy's poly1d.
        out.append(case(f"pade/{i}", "pade", [s, m, n], {"p": p[::-1], "q": q[::-1]}, rtol=1e-10, atol=1e-12))
    return out
