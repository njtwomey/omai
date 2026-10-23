"""Shared helpers of the optim case modules: tight cvxpy solves, histograms and point clouds (oracle-core group)."""

from collections.abc import Sequence
from typing import cast

import cvxpy as cp
import numpy as np

from whitebox_oracle.helpers import FloatArray, rng


def cvx_minimize(objective: object, x: cp.Variable, constraints: Sequence[object] = ()) -> tuple[FloatArray, float]:
    """Minimise with Clarabel at tight tolerances; return the minimiser and the optimal value.

    cvxpy types its operator overloads loosely, so expressions and constraints arrive as objects and are cast here.
    """
    problem = cp.Problem(cp.Minimize(cast(cp.Expression, objective)), cast(list[cp.Constraint], list(constraints)))
    value = problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(x.value, dtype=np.float64), float(cast(float, value))


def histogram(g: np.random.Generator, n: int) -> FloatArray:
    """A random histogram with entries bounded away from zero, summing to one."""
    h = g.uniform(0.2, 1.0, size=n)
    return h / h.sum()


def clouds(seed: int, n: int, m: int, d: int) -> tuple[FloatArray, FloatArray]:
    """Two Gaussian point clouds in R^d, the second shifted by one in every coordinate."""
    g = rng(seed)
    return g.normal(size=(n, d)), g.normal(size=(m, d)) + 1.0


def sparse_problem(seed: int, m: int, k: int, nnz: int, noise: float) -> tuple[FloatArray, FloatArray]:
    """A Gaussian design [m, k] and the response of an nnz-sparse coefficient vector plus noise."""
    g = rng(seed)
    d = g.normal(size=(m, k))
    x = np.zeros(k)
    x[g.choice(k, size=nnz, replace=False)] = g.normal(size=nnz) * 2
    return d, d @ x + g.normal(size=m) * noise
