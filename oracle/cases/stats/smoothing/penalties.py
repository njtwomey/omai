"""Penalties against numpy (difference penalties), scipy (derivative penalties of B-splines integrated with
scipy.integrate.quad on each knot span) and scipy.linalg.qr (the sum-to-zero constraint as patsy absorbs it)."""

import numpy as np
import scipy
from scipy import linalg
from scipy.integrate import quad
from scipy.interpolate import BSpline
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def product_integral(f: BSpline, g: BSpline, a: float, b: float) -> float:
    def integrand(u: float) -> float:
        return float(f(u) * g(u))

    return float(quad(integrand, a, b, epsabs=0, epsrel=1e-13)[0])


def derivative_penalty(t: FloatArray, k: int, m: int) -> FloatArray:
    n = t.size - k - 1
    basis = [BSpline(t, np.eye(n)[i], k).derivative(m) for i in range(n)]
    S = np.zeros((n, n))
    spans = [float(v) for v in np.unique(t[k : n + 1])]
    for i in range(n):
        for j in range(n):
            S[i, j] = sum(
                product_integral(basis[i], basis[j], a, b) for a, b in zip(spans[:-1], spans[1:], strict=True)
            )
    return S


def cases() -> list[Case]:
    out: list[Case] = []
    for k, order in [(6, 1), (6, 2), (8, 3)]:
        D = np.diff(np.eye(k), n=order, axis=0)
        out.append(case(f"difference/{k}/{order}", "difference", [], D.T @ D, k=k, order=order))
    t = np.array([0, 0, 0, 0, 1.5, 3.0, 4.0, 6, 6, 6, 6], dtype=float)
    out.append(
        case(
            "derivative/cubic/2",
            "derivative",
            [t, t],
            derivative_penalty(t, 3, 2),
            rtol=1e-9,
            atol=1e-11,
            degree=3,
            order=2,
        )
    )
    out.append(
        case(
            "derivative/cubic/1",
            "derivative",
            [t, t],
            derivative_penalty(t, 3, 1),
            rtol=1e-9,
            atol=1e-11,
            degree=3,
            order=1,
        )
    )
    X = rng(5).uniform(size=(20, 5))
    C = X.mean(0, keepdims=True)
    q, _ = linalg.qr(C.T)
    out.append(case("sumToZero", "sumToZero", [X, X], np.asarray(q[:, 1:]), rtol=1e-10, atol=1e-12))
    C2 = rng(6).normal(size=(2, 5))
    q2, _ = linalg.qr(C2.T)
    out.append(
        case("constraintNullSpace", "constraintNullSpace", [C2, C2], np.asarray(q2[:, 2:]), rtol=1e-10, atol=1e-12)
    )
    return out
