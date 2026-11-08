"""Triangular solves against scipy.linalg.solve_triangular."""

import numpy as np
import scipy
import scipy.linalg as sl
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, well_conditioned

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(20)
    a = well_conditioned(g, 4, 4)
    b = g.normal(size=(4, 3))
    out: list[Case] = []
    for lower in (True, False):
        for trans in (False, True):
            for unit in (False, True):
                t = np.tril(a) if lower else np.triu(a)
                x = sl.solve_triangular(t, b, lower=lower, trans=1 if trans else 0, unit_diagonal=unit)
                flags = ("lower" if lower else "upper") + ("/trans" if trans else "") + ("/unit" if unit else "")
                name = f"solveTriangular/{flags}"
                out.append(
                    case(
                        name, "solveTriangular", [t, b], x, rtol=1e-10, lower=lower, transpose=trans, unitDiagonal=unit
                    )
                )
    vec = g.normal(size=4)
    out.append(
        case(
            "solveTriangular/vector",
            "solveTriangular",
            [np.tril(a), vec],
            sl.solve_triangular(np.tril(a), vec, lower=True),
            rtol=1e-10,
            lower=True,
        )
    )
    batch = np.tril(well_conditioned(g, 3, 3, (2,)))
    bb = g.normal(size=(2, 3, 2))
    out.append(
        case(
            "solveTriangular/batch",
            "solveTriangular",
            [batch, bb],
            np.linalg.solve(batch, bb),  # batch is lower triangular, so a general batched solve is the reference
            rtol=1e-10,
            lower=True,
        )
    )
    return out
