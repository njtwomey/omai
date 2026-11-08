"""Further decompositions against numpy and scipy: eigenvalues of general matrices, Hessenberg and Schur invariants,
packed LU, rank-one Cholesky updates, null spaces, polar decompositions and orthonormal bases."""

import numpy as np
import scipy
import scipy.linalg as sl
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def sorted_eigenvalues(a: FloatArray) -> FloatArray:
    w = np.linalg.eigvals(a)
    order = np.lexsort((w.imag, w.real))
    return np.stack([w.real[order], w.imag[order]])


def cases() -> list[Case]:
    g = rng(28)
    out: list[Case] = []
    for n in [3, 5, 8]:
        a = g.normal(size=(n, n))
        out.append(case(f"eigvals/{n}", "eigvalsSorted", [a], sorted_eigenvalues(a), rtol=1e-9, atol=1e-11))
    rot = np.array([[0.0, -2.0, 1.0], [2.0, 0.0, 0.5], [0.0, 0.0, 3.0]])
    out.append(case("eigvals/complex", "eigvalsSorted", [rot], sorted_eigenvalues(rot), rtol=1e-10, atol=1e-12))
    a = g.normal(size=(5, 5))
    lu, piv = sl.lu_factor(a)
    b = g.normal(size=(5, 2))
    out.append(case("luFactor/lu", "luFactor", [a], lu, rtol=1e-10, atol=1e-14, part="lu"))
    out.append(case("luFactor/piv", "luFactor", [a], piv.astype(np.float64), part="piv"))
    out.append(case("luSolve", "luSolve", [a, b], sl.lu_solve((lu, piv), b), rtol=1e-10))
    out.append(
        case("luSolve/transpose", "luSolve", [a, b], sl.lu_solve((lu, piv), b, trans=1), rtol=1e-10, transpose=True)
    )
    s = spd(g, 4)
    x = g.normal(size=4) * 0.5
    out.append(
        case(
            "choleskyUpdate",
            "choleskyUpdate",
            [np.linalg.cholesky(s), x],
            np.linalg.cholesky(s + np.outer(x, x)),
            rtol=1e-10,
        )
    )
    out.append(
        case(
            "choleskyDowndate",
            "choleskyDowndate",
            [np.linalg.cholesky(s), x],
            np.linalg.cholesky(s - np.outer(x, x)),
            rtol=1e-9,
        )
    )
    rect = g.normal(size=(5, 3))
    u, p = sl.polar(rect)
    out.append(case("polar/u", "polar", [rect], u, rtol=1e-10, atol=1e-13, part="u"))
    out.append(case("polar/p", "polar", [rect], p, rtol=1e-10, atol=1e-13, part="p"))
    sq = g.normal(size=(4, 4))
    _, pl = sl.polar(sq, side="left")
    out.append(case("polar/left/p", "polar", [sq], pl, rtol=1e-10, atol=1e-13, part="p", side="left"))
    low = g.normal(size=(5, 2)) @ g.normal(size=(2, 4))
    # Null spaces and ranges are compared as projectors (bases are not unique).
    ns = sl.null_space(low)
    out.append(case("nullSpace/projector", "nullSpaceProjector", [low], ns @ ns.T, rtol=1e-9, atol=1e-10))
    rb = sl.orth(low)
    out.append(case("orth/projector", "orthProjector", [low], rb @ rb.T, rtol=1e-9, atol=1e-10))
    return out
