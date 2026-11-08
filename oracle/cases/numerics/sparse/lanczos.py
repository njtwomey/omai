"""The stepped Lanczos process against a dense numpy implementation with full reorthogonalisation and scipy.linalg."""

import numpy as np
import scipy
import scipy.linalg as sla
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def lanczos(a: FloatArray, v0: FloatArray, m: int) -> tuple[FloatArray, FloatArray]:
    """Alpha (m) and beta (m, the last the residual coupling) of m Lanczos steps with full reorthogonalisation."""
    q = [v0 / np.linalg.norm(v0)]
    alpha: list[float] = []
    beta: list[float] = []
    for j in range(m):
        w = a @ q[j]
        alpha.append(float(w @ q[j]))
        for _ in range(2):
            for b in q:
                w = w - (w @ b) * b
        beta.append(float(np.linalg.norm(w)))
        q.append(w / beta[-1])
    return np.array(alpha), np.array(beta)


def tridiagonal(alpha: FloatArray, beta: FloatArray) -> FloatArray:
    m = len(alpha)
    return np.diag(alpha) + np.diag(beta[: m - 1], 1) + np.diag(beta[: m - 1], -1)


def cases() -> list[Case]:
    g = rng(140)
    n = 12
    raw = g.normal(size=(n, n))
    a = (raw + raw.T) / 2
    v0 = g.normal(size=n)
    out: list[Case] = []
    alpha, beta = lanczos(a, v0, 6)
    out.append(case("alpha", "alpha", [a, v0], alpha, rtol=1e-9, atol=1e-10, steps=6))
    out.append(case("beta", "beta", [a, v0], beta, rtol=1e-9, atol=1e-10, steps=6))
    out.append(case("ritz/6", "ritz", [a, v0], sla.eigvalsh(tridiagonal(alpha, beta)), rtol=1e-9, atol=1e-10, steps=6))
    # After n steps the Krylov space is the whole space: the Ritz values are the eigenvalues.
    out.append(case("ritz/full", "ritz", [a, v0], sla.eigvalsh(a), rtol=1e-8, atol=1e-9, steps=n))
    return out
