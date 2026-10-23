"""Sparse matrices, Krylov solvers and sparse eigensolvers against scipy.sparse and scipy.sparse.linalg."""

from typing import Literal

import numpy as np
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as sla
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

WHICH: list[Literal["LM", "LA", "SA"]] = ["LM", "LA", "SA"]


def laplacian(n: int) -> sps.csr_array:
    """The 2-D five-point Laplacian on an n x n grid (symmetric positive definite)."""
    one = sps.csr_array(2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    eye = sps.eye_array(n)
    return sps.csr_array(sps.kron(one, eye) + sps.kron(eye, one))


def dense(m: sps.csr_array) -> FloatArray:
    return np.asarray(m.toarray(), dtype=np.float64)


def cases() -> list[Case]:
    g = rng(110)
    out: list[Case] = []
    rows = np.array([0, 2, 1, 0, 2, 2, 1], dtype=np.float64)
    cols = np.array([1, 0, 1, 1, 2, 0, 3], dtype=np.float64)
    vals = g.normal(size=7)
    coo = sps.coo_array((vals, (rows.astype(int), cols.astype(int))), shape=(3, 4)).tocsr()
    coo.sum_duplicates()
    coo.sort_indices()
    out.append(case("fromCoo/dense", "fromCooDense", [rows, cols, vals], dense(coo), rtol=1e-14))
    out.append(
        case("fromCoo/data", "fromCooData", [rows, cols, vals], np.asarray(coo.data, dtype=np.float64), rtol=1e-14)
    )
    x = g.normal(size=4)
    out.append(case("spmv", "spmv", [rows, cols, vals, x], coo @ x, rtol=1e-13))
    bm = g.normal(size=(4, 2))
    out.append(case("spmm", "spmm", [rows, cols, vals, bm], coo @ bm, rtol=1e-13))
    out.append(case("transpose", "transpose", [rows, cols, vals], dense(sps.csr_array(coo.T)), rtol=1e-14))

    spd = laplacian(6)
    spd = sps.csr_array(spd + sps.diags_array(g.uniform(0, 0.5, size=36)))
    nonsym = sps.csr_array(
        spd + 0.3 * sps.diags_array(np.ones(35), offsets=1) - 0.1 * sps.diags_array(np.ones(35), offsets=-1)
    )
    indefinite = sps.csr_array(spd - 2.5 * sps.eye_array(36))
    b = g.normal(size=36)
    for name, matrix, methods in [
        ("spd", spd, ["cg", "bicgstab", "gmres", "minres"]),
        ("nonsymmetric", nonsym, ["bicgstab", "gmres"]),
        ("indefinite", indefinite, ["minres", "gmres"]),
    ]:
        exact = sla.spsolve(sps.csc_array(matrix), b)
        for method in methods:
            out.append(
                case(
                    f"{method}/{name}",
                    "solve",
                    [dense(matrix), b],
                    np.asarray(exact, dtype=np.float64),
                    rtol=1e-8,
                    atol=1e-9,
                    method=method,
                )
            )
    sym = laplacian(8)
    for which in WHICH:
        w = sla.eigsh(sym, k=3, which=which, tol=1e-13, v0=np.ones(64))[0]
        out.append(
            case(
                f"eigsh/{which}",
                "eigsh",
                [dense(sym)],
                np.sort(np.asarray(w, dtype=np.float64)),
                rtol=1e-9,
                atol=1e-10,
                which=which,
            )
        )
    rect = sps.random_array((40, 25), density=0.2, rng=g, format="csr")
    s = sla.svds(rect, k=3, tol=1e-13, v0=np.ones(25))[1]
    out.append(
        case(
            "svds", "svds", [dense(sps.csr_array(rect))], np.sort(np.asarray(s, dtype=np.float64)), rtol=1e-8, atol=1e-9
        )
    )
    return out
