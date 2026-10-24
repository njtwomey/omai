"""Finite-difference solves against scipy.sparse.linalg.spsolve on the same discretisations.

The 5-point Laplacian with Dirichlet data is built with scipy.sparse (Kronecker sums) and solved directly; the
TypeScript side solves the same system with its iterations (CG, SOR, multigrid) and must agree to the iteration
tolerance. One implicit Crank-Nicolson heat run (1-D, Dirichlet) is compared step for step.
"""

import numpy as np
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def second_difference(m: int) -> sps.csc_array:
    """The tridiagonal [1, -2, 1] matrix of order m."""
    return sps.csc_array(np.diag(np.ones(m - 1), -1) - 2 * np.eye(m) + np.diag(np.ones(m - 1), 1))


def source(x: FloatArray, y: FloatArray) -> FloatArray:
    return np.sin(np.pi * x) * np.cos(2 * y) + x * y


def boundary(x: FloatArray, y: FloatArray) -> FloatArray:
    return x**2 - y


def poisson2d(n: int) -> FloatArray:
    """Delta u = f on [0,1]^2 with u = g on the boundary, n nodes per side; u[i, j] at (x_i, y_j)."""
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    m = n - 2
    d = second_difference(m) / h**2
    eye = sps.eye_array(m)
    a = sps.kron(d, eye) + sps.kron(eye, d)
    g = boundary(xx, yy)
    rhs = source(xx, yy)[1:-1, 1:-1].copy()
    rhs[0, :] -= g[0, 1:-1] / h**2
    rhs[-1, :] -= g[-1, 1:-1] / h**2
    rhs[:, 0] -= g[1:-1, 0] / h**2
    rhs[:, -1] -= g[1:-1, -1] / h**2
    inner = spla.spsolve(sps.csc_array(a), rhs.ravel())
    u = g.copy()
    u[1:-1, 1:-1] = np.asarray(inner).reshape(m, m)
    return u


def crank_nicolson(n: int, dt: float, steps: int) -> FloatArray:
    """u_t = u_xx on [0, 1], u(0) = 0, u(1) = 1, from u0 = sin(pi x) + x."""
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    m = n - 2
    lap = second_difference(m) / h**2
    b = np.zeros(m)
    b[-1] = 1.0 / h**2
    eye = sps.eye_array(m)
    lhs = sps.csc_array(eye - 0.5 * dt * lap)
    rhs_op = eye + 0.5 * dt * lap
    u = (np.sin(np.pi * x) + x)[1:-1]
    for _ in range(steps):
        u = np.asarray(spla.spsolve(lhs, rhs_op @ u + dt * b))
    return np.concatenate([[0.0], u, [1.0]])


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [9, 17]:
        u = poisson2d(n)
        for method in ["cg", "sor", "multigrid", "gauss-seidel"]:
            out.append(case(f"poisson2d/{n}/{method}", "poisson2d", [], u, rtol=1e-7, atol=1e-8, n=n, method=method))
    out.append(case("heat/crank-nicolson", "heat_cn", [], crank_nicolson(21, 0.01, 10), rtol=1e-11, atol=1e-12))
    return out
