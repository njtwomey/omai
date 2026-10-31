"""Subspace iteration and one-sided Jacobi SVD against dense numpy / scipy.linalg references."""

import numpy as np
import scipy
import scipy.linalg as sla
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def subspace(a: FloatArray, x0: FloatArray, steps: int) -> tuple[FloatArray, FloatArray]:
    """The projector onto the block after `steps` iterations and its Ritz values by decreasing magnitude."""
    q, _ = np.linalg.qr(x0)
    for _ in range(steps):
        q, _ = np.linalg.qr(a @ q)
    w = np.linalg.eigvalsh(q.T @ a @ q)
    return q @ q.T, w[np.argsort(-np.abs(w))]


def cases() -> list[Case]:
    g = rng(150)
    out: list[Case] = []
    n, k = 10, 3
    basis, _ = np.linalg.qr(g.normal(size=(n, n)))
    spectrum = np.array([10.0, -8.0, 6.0, 3.0, 2.5, 2.0, 1.0, 0.5, 0.2, 0.1])
    a = basis @ np.diag(spectrum) @ basis.T
    a = (a + a.T) / 2
    x0 = g.normal(size=(n, k))
    for steps in (1, 5, 25):
        projector, ritz = subspace(a, x0, steps)
        out.append(case(f"subspace/{steps}/projector", "projector", [a, x0], projector, rtol=1e-9, atol=1e-10, steps=steps))
        out.append(case(f"subspace/{steps}/ritz", "ritz", [a, x0], ritz, rtol=1e-9, atol=1e-10, steps=steps))
    eigenvalues = np.linalg.eigvalsh(a)
    out.append(
        case("subspace/converged", "converged", [a, x0], eigenvalues[np.argsort(-np.abs(eigenvalues))][:k], rtol=1e-8, atol=1e-9)
    )

    tall = g.normal(size=(8, 5)) @ np.diag([3.0, 1.0, 0.3, 0.05, 0.001])
    out.append(case("jacobi/values", "values", [tall], sla.svdvals(tall), rtol=1e-10, atol=1e-13))
    out.append(case("jacobi/reconstruct", "reconstruct", [tall], tall, rtol=1e-12, atol=1e-13))
    square = g.normal(size=(6, 6))
    out.append(case("jacobi/square", "values", [square], sla.svdvals(square), rtol=1e-11, atol=1e-13))
    return out
