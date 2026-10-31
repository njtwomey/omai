"""Stationary iterations and the two-grid cycle against dense numpy / scipy.linalg reference computations."""

import numpy as np
import scipy
import scipy.linalg as sla
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def iteration_matrix(a: FloatArray, method: str, omega: float) -> tuple[FloatArray, FloatArray]:
    """G and the map c = M b of x_{k+1} = G x_k + M b for the splitting of `method`, from dense triangular parts."""
    d = np.diag(np.diag(a))
    lower = np.tril(a, -1)
    upper = np.triu(a, 1)
    if method == "jacobi":
        m_inv = omega * np.linalg.inv(d)
        return np.eye(len(a)) - m_inv @ a, m_inv
    w = 1.0 if method == "gauss-seidel" else omega
    lhs = d + w * lower
    return np.linalg.solve(lhs, (1 - w) * d - w * upper), w * np.linalg.inv(lhs)


def iterate(a: FloatArray, b: FloatArray, method: str, omega: float, steps: int) -> FloatArray:
    g, m = iteration_matrix(a, method, omega)
    x = np.zeros(len(b))
    for _ in range(steps):
        x = g @ x + m @ b
    return x


def poisson(n: int) -> FloatArray:
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def two_grid(a: FloatArray, b: FloatArray, cycles: int, omega: float) -> FloatArray:
    """Damped-Jacobi two-grid cycles with linear interpolation and the Galerkin coarse operator (dense)."""
    n = len(b)
    nc = (n - 1) // 2
    p = np.zeros((n, nc))
    for j in range(nc):
        p[2 * j + 1, j] = 1.0
        p[2 * j, j] = 0.5
        p[2 * j + 2, j] = 0.5
    ac = p.T @ a @ p
    d = np.diag(a)
    x = np.zeros(n)
    for _ in range(cycles):
        x = x + omega * (b - a @ x) / d
        x = x + p @ np.linalg.solve(ac, p.T @ (b - a @ x))
        x = x + omega * (b - a @ x) / d
    return x


def cases() -> list[Case]:
    g = rng(130)
    out: list[Case] = []
    n = 9
    noise = g.normal(size=(n, n))
    a = poisson(n) + 0.3 * (noise + noise.T) / 2 + 0.5 * np.eye(n)
    b = g.normal(size=n)
    for method, omega in [("jacobi", 1.0), ("jacobi", 2 / 3), ("gauss-seidel", 1.0), ("sor", 1.3), ("sor", 0.8)]:
        tag = f"{method}/{omega:.2f}"
        gm, _ = iteration_matrix(a, method, omega)
        out.append(
            case(
                f"{tag}/iterate",
                "iterate",
                [a, b],
                iterate(a, b, method, omega, 7),
                method=method,
                omega=omega,
                steps=7,
            )
        )
        out.append(case(f"{tag}/matrix", "matrix", [a], gm, method=method, omega=omega))
        radius = float(np.max(np.abs(np.linalg.eigvals(gm))))
        out.append(case(f"{tag}/radius", "radius", [a], np.array(radius), rtol=1e-10, method=method, omega=omega))
    # Convergence to the direct solution, on a diagonally dominant system every method converges for.
    dom = a + 4 * np.eye(n)
    exact = sla.solve(dom, b)
    for method, omega in [("jacobi", 1.0), ("gauss-seidel", 1.0), ("sor", 1.2)]:
        out.append(
            case(f"{method}/solution", "solution", [dom, b], exact, rtol=1e-9, atol=1e-10, method=method, omega=omega)
        )
    # The two-grid cycle on 1-D Poisson: the iterate after a few cycles, and the converged solution.
    m = 15
    pa = poisson(m)
    pb = g.normal(size=m)
    out.append(
        case("twoGrid/iterate", "twoGrid", [pa, pb], two_grid(pa, pb, 3, 2 / 3), rtol=1e-10, atol=1e-12, cycles=3)
    )
    out.append(case("twoGrid/solution", "twoGridSolution", [pa, pb], sla.solve(pa, pb), rtol=1e-8, atol=1e-9))
    return out
