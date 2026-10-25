"""Step-by-step linear algebra against sympy (rref, nullspace, columnspace, rowspace, eigenvalues), scipy.linalg (lu)
and numpy (det, poly, lstsq, svd, eigh)."""

from typing import cast

import numpy as np
import scipy
import scipy.linalg as sl
import sympy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}


def ints(g: np.random.Generator, m: int, n: int) -> FloatArray:
    return np.asarray(g.integers(-4, 5, size=(m, n)), dtype=np.float64)


def to_sympy(a: FloatArray) -> sympy.Matrix:
    return sympy.Matrix([[sympy.Integer(int(v)) for v in row] for row in a])


def to_numpy(m: sympy.Matrix) -> FloatArray:
    return np.array([[float(sympy.N(v, 20)) for v in row] for row in m.tolist()], dtype=np.float64).reshape(m.shape)


def vectors(vs: list[sympy.Matrix], length: int) -> FloatArray:
    """Column vectors as rows of an array (empty bases have shape [0, length])."""
    return np.array([to_numpy(v).ravel() for v in vs], dtype=np.float64).reshape(len(vs), length)


def cases() -> list[Case]:
    g = rng(1208)
    out: list[Case] = []
    shapes = [(3, 3), (3, 4), (4, 3), (3, 5), (4, 4)]
    for i, (m, n) in enumerate(shapes):
        a = ints(g, m, n)
        if i == 4:
            a[3] = a[0] + a[1]
        if i == 1:
            a[2] = 2 * a[0] - a[1]
        sa = to_sympy(a)
        reduced, pivots = cast(tuple[sympy.Matrix, tuple[int, ...]], sa.rref())
        out.append(case(f"rref/{i}", "rref", [a], to_numpy(reduced), rtol=1e-12, atol=1e-12))
        out.append(case(f"pivots/{i}", "pivotColumns", [a], np.array(pivots, dtype=np.int64)))
        out.append(case(f"rank/{i}", "rank", [a], np.array(sa.rank())))
        out.append(case(f"nullspace/{i}", "nullSpace", [a], vectors(sa.nullspace(), n), rtol=1e-12, atol=1e-12))
        out.append(case(f"columnspace/{i}", "columnSpace", [a], vectors(sa.columnspace(), m), atol=1e-12))
        out.append(case(f"leftnull/{i}", "leftNullSpace", [a], vectors(sa.T.nullspace(), m), rtol=1e-12, atol=1e-12))
    for i in range(4):
        a = g.normal(size=(4, 4))
        p, low, up = sl.lu(a)
        out.append(case(f"lu/{i}/p", "luP", [a], p))
        out.append(case(f"lu/{i}/l", "luL", [a], low, rtol=1e-12, atol=1e-14))
        out.append(case(f"lu/{i}/u", "luU", [a], up, rtol=1e-12, atol=1e-14))
        out.append(case(f"echelon/{i}", "echelonPartial", [a], up, rtol=1e-12, atol=1e-14))
        out.append(case(f"det/elimination/{i}", "detByElimination", [a], np.array(np.linalg.det(a)), rtol=1e-12))
        out.append(case(f"poly/{i}", "characteristicPolynomial", [a], np.poly(a), rtol=1e-10, atol=1e-12))
    for i, size in enumerate([3, 4, 5]):
        a = ints(g, size, size)
        out.append(
            case(
                f"det/cofactors/{i}",
                "detByCofactors",
                [a],
                np.array(float(sympy.N(cast(sympy.Expr, to_sympy(a).det()), 20))),
                atol=1e-9,
            )
        )
    for i in range(3):
        a = ints(g, 3, 4)
        b = ints(g, 3, 1).ravel()
        sa, sb = to_sympy(a), sympy.Matrix(b.astype(int).tolist())
        sol, params = cast(tuple[sympy.Matrix, sympy.Matrix], sa.gauss_jordan_solve(sb))
        particular = sol.subs({p: 0 for p in params})
        out.append(case(f"solve/{i}", "particular", [a, b], to_numpy(particular).ravel(), rtol=1e-12, atol=1e-12))
    for i in range(3):
        a = g.normal(size=(6, 3))
        b = g.normal(size=6)
        x = np.linalg.lstsq(a, b, rcond=None)[0]
        out.append(case(f"lstsq/{i}", "leastSquares", [a, b], x, rtol=1e-10))
        out.append(case(f"lstsq/{i}/residual", "leastSquaresResidual", [a, b], b - a @ x, rtol=1e-9, atol=1e-12))
        out.append(case(f"projection/{i}", "projectionMatrix", [a], a @ np.linalg.pinv(a), rtol=1e-10, atol=1e-12))
    for i, a in enumerate(
        [np.array([[2.0, 1.0], [1.0, 3.0]]), g.normal(size=(3, 3)) + 3 * np.eye(3), np.diag([1.0, 2.0, 2.0])]
    ):
        w = np.linalg.eigvals(a)
        order = np.lexsort((w.imag, w.real))
        out.append(case(f"eigen/{i}/re", "eigenRe", [a], w.real[order], rtol=1e-8, atol=1e-10))
        out.append(case(f"eigen/{i}/im", "eigenIm", [a], np.abs(w.imag[order]), rtol=1e-8, atol=1e-10))
    for i in range(3):
        a = g.normal(size=(3, 3))
        sym = a + a.T
        w, v = np.linalg.eigh(sym)
        out.append(case(f"principal/{i}", "principalValues", [sym], w, rtol=1e-10, atol=1e-12))
        x = g.normal(size=(4, 3))
        out.append(case(f"quadratic/{i}", "quadraticForm", [sym, x], np.einsum("ni,ij,nj->n", x, sym, x), rtol=1e-12))
        s = np.linalg.svd(a, compute_uv=False)
        out.append(case(f"svd/{i}", "singularValues", [a], s, rtol=1e-10))
        # The semi-axes of the image of the unit sphere have lengths sigma_i.
        out.append(case(f"svd/{i}/axes", "svdAxisLengths", [a], s, rtol=1e-10))
        del v
    return out
