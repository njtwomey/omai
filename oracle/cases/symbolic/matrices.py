"""Symbolic matrices against SymPy's Matrix: determinants and characteristic polynomials (exact text), inverses (by
value, entry by entry), and eigenvalues (sorted numeric values with multiplicities)."""

from typing import cast

import sympy as sp
from whitebox_oracle.fixture import Case, Json, case
from whitebox_oracle.symbolic import expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}


MATRICES: list[list[list[str]]] = [
    [["a", "b"], ["c", "d"]],
    [["1", "2"], ["3", "4"]],
    [["2", "1", "0"], ["1", "2", "1"], ["0", "1", "2"]],
    [["x", "1", "0"], ["0", "x", "1"], ["1", "0", "x"]],
    [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "10"]],
    [["2", "0", "0", "1"], ["0", "3", "1", "0"], ["0", "1", "3", "0"], ["1", "0", "0", "2"]],
    [["cos(t)", "-sin(t)"], ["sin(t)", "cos(t)"]],
]

EIGEN: list[list[list[str]]] = [
    [["2", "1"], ["1", "2"]],
    [["1", "2"], ["3", "4"]],
    [["2", "1", "0"], ["1", "2", "1"], ["0", "1", "2"]],
    [["0", "-1"], ["1", "0"]],
    [["3", "1"], ["0", "3"]],
    [["2", "0", "0", "1"], ["0", "3", "1", "0"], ["0", "1", "3", "0"], ["1", "0", "0", "2"]],
    [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "10"]],
]


def to_matrix(rows: list[list[str]]) -> sp.Matrix:
    return sp.Matrix([[parse(v) for v in r] for r in rows])


def eigen_values(m: sp.Matrix) -> list[Json]:
    vals: list[complex] = []
    for v, k in cast(dict[sp.Expr, int], m.eigenvals()).items():
        vals.extend([complex(v)] * int(k))
    keyed = sorted(vals, key=lambda z: (abs(z.imag) > 1e-12, z.real, z.imag))
    return [[z.real, 0.0 if abs(z.imag) < 1e-12 else z.imag] for z in keyed]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, rows in enumerate(MATRICES):
        m = to_matrix(rows)
        src = ";".join(",".join(r) for r in rows)
        d = cast(sp.Expr, m.det())
        out.append(expr_case(f"det/{i}", "det", [src], sp.expand(d)))
        out.append(expr_case(f"charpoly/{i}", "charpoly", [src], m.charpoly("lam").as_expr()))
        if sp.simplify(d) != 0:
            inv = m.inv()
            for r in range(m.rows):
                for c in range(m.cols):
                    entry = cast(sp.Expr, inv[r, c])
                    out.append(expr_case(f"inverse/{i}/{r}{c}", "inverse", [src, f"{r},{c}"], entry, exact=False))
    for i, rows in enumerate(EIGEN):
        m = to_matrix(rows)
        src = ";".join(",".join(r) for r in rows)
        out.append(case(f"eigenvals/{i}", "eigenvals", [src], eigen_values(m), rtol=1e-9, atol=1e-9))
    return out
