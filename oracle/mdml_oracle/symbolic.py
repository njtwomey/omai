"""Shared pieces for the symbolic case modules: parsing SymPy-syntax strings with the usual symbols, and evaluating
results at fixed sample points with numpy (complex arithmetic, principal branches) so that the TypeScript tests can
compare values as well as printed forms."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.special as sps
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .fixture import Case, Json, case

type ComplexArray = npt.NDArray[np.complex128]
"""Values at the sample points."""

NAMES: tuple[str, ...] = ("x", "y", "z", "a", "b", "c", "t", "p", "q", "r", "s", "k", "n", "u", "v", "w", "lam")
SYMBOLS: dict[str, sp.Symbol] = {n: sp.Symbol(n) for n in NAMES}
"""Plain (complex) symbols, as `sym(name)` in mdml-js."""

# Sample points for up to four symbols: each symbol gets its own row, chosen away from the usual singularities.
POINTS: list[list[float]] = [
    [0.37, 0.71, 1.23, 1.9, 2.6],
    [0.53, 1.07, 0.29, 1.61, 2.2],
    [1.31, 0.43, 0.87, 2.3, 0.61],
    [0.77, 1.49, 0.23, 1.13, 1.97],
]


def point_row(i: int) -> list[float]:
    """Sample values of the i-th symbol (in sorted name order): a base row, shifted by 0.1 for every four symbols."""
    return [v + 0.1 * (i // len(POINTS)) for v in POINTS[i % len(POINTS)]]


def parse_any(text: str, local: Mapping[str, object] | None = None) -> object:
    """Parse SymPy syntax with the plain symbols (and `local`) bound by name; the result may be a tuple or relation."""
    names: dict[str, Any] = {**SYMBOLS, **(local or {})}
    return cast(object, parse_expr(text, local_dict=names))


def parse(text: str, local: Mapping[str, object] | None = None) -> sp.Expr:
    """Parse SymPy syntax into an expression."""
    e = parse_any(text, local)
    if not isinstance(e, sp.Expr):
        raise TypeError(f"{text!r} is not an expression")
    return e


def to_text(e: sp.Basic) -> str:
    """SymPy's `str()`, the form mdml-js's `toText` reproduces."""
    return str(e)


def free_names(e: sp.Basic) -> list[str]:
    """Names of the free symbols, sorted."""
    return sorted(str(s) for s in e.free_symbols)


def input_names(text: str) -> list[str]:
    """Free symbol names of an input in SymPy syntax; none when the text is not SymPy syntax (implicit products)."""
    try:
        e = parse_any(text)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return []
    return free_names(e) if isinstance(e, sp.Basic) else []


def _evaluate(e: sp.Basic, names: Sequence[str], dtype: type[np.floating] | type[np.complexfloating]) -> ComplexArray:
    """e at the sample points with real or complex arithmetic; NaN where numpy cannot evaluate it."""
    syms = [SYMBOLS.get(n, sp.Symbol(n)) for n in names]
    special = {"erf": sps.erf, "gamma": sps.gamma, "polygamma": sps.polygamma, "factorial": _factorial}
    ones = np.ones(len(POINTS[0]), dtype=np.complex128)
    try:
        f = sp.lambdify(syms, e, modules=[special, "numpy"])
    except Exception:  # unevaluated integrals, derivatives and orders have no numeric form
        return ones * np.nan
    cols = [np.asarray(point_row(i), dtype=dtype) for i in range(len(names))]
    with np.errstate(all="ignore"):
        try:
            return np.asarray(f(*cols), dtype=np.complex128) * ones
        except (TypeError, ValueError, AttributeError, NameError):
            return ones * np.nan


def values(e: sp.Basic, names: Sequence[str]) -> list[Json]:
    """[re, im] of e at the sample points (symbol i takes point_row(i)): real arithmetic where it is defined, complex
    arithmetic (principal branches) elsewhere; non-finite values are encoded as strings."""
    real = _evaluate(e, names, np.float64)
    complex_ = _evaluate(e, names, np.complex128)
    out = np.where(np.isfinite(real), real, complex_)
    return [[_num(float(z.real)), _num(float(z.imag))] for z in out]


def _factorial(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """x! as gamma(x + 1), so that it accepts complex arrays."""
    return np.asarray(sps.gamma(v + 1), dtype=np.complex128)


def _num(v: float) -> Json:
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def expr_case(
    name: str, fn: str, inputs: Sequence[str], result: sp.Basic, *, exact: bool = True, **params: Json
) -> Case:
    """A case whose output is a SymPy expression: its text, LaTeX and values at the sample points.

    `exact` says whether mdml-js should reproduce SymPy's text; otherwise only the values are compared.
    """
    names = sorted(set(free_names(result)) | {n for s in inputs for n in input_names(s)})
    return case(
        name,
        fn,
        list(inputs),
        {"text": to_text(result), "latex": sp.latex(result), "values": values(result, names), "names": list(names)},
        rtol=1e-9,
        atol=1e-9,
        exact=exact,
        **params,
    )
