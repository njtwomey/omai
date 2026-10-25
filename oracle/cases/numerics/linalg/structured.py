"""Structured matrices and solves against scipy.linalg."""

import numpy as np
import scipy
import scipy.linalg as sl
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(27)
    c, r = g.normal(size=4), g.normal(size=3)
    b = g.normal(size=4)
    n = 5
    lower, diag, upper = g.normal(size=n - 1), g.normal(size=n) + 4, g.normal(size=n - 1)
    dense = np.diag(diag) + np.diag(upper, 1) + np.diag(lower, -1)
    rhs = g.normal(size=n)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1] = diag
    ab[2, :-1] = lower
    tc = np.array([4.0, 1.0, 0.5, 0.2])
    a, bb = g.normal(size=(2, 2)) + 3 * np.eye(2), g.normal(size=(3, 3)) + 3 * np.eye(3)
    kc = g.normal(size=6)
    return [
        case("toeplitz", "toeplitz", [c, r], sl.toeplitz(c, r)),
        case("toeplitz/symmetric", "toeplitz", [c], sl.toeplitz(c)),
        case("circulant", "circulant", [c], sl.circulant(c)),
        case("hankel", "hankel", [c, r], sl.hankel(c, r)),
        case("hankel/default", "hankel", [c], sl.hankel(c)),
        case("blockDiag", "blockDiag", [a, bb], sl.block_diag(a, bb)),
        case(
            "solveTridiagonal", "solveTridiagonal", [lower, diag, upper, rhs], np.linalg.solve(dense, rhs), rtol=1e-10
        ),
        case("solveBanded", "solveBanded", [ab, rhs], sl.solve_banded((1, 1), ab, rhs), rtol=1e-10, l=1, u=1),
        case("solveToeplitz", "solveToeplitz", [tc, b], sl.solve_toeplitz(tc, b), rtol=1e-10),
        case("kroneckerSolve", "kroneckerSolve", [a, bb, kc], np.linalg.solve(np.kron(a, bb), kc), rtol=1e-10),
    ]
