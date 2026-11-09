"""Analytical nonlinear (quadratic-inverse) shrinkage (covariance/nonlinear-shrinkage.ts). No installed library has it
(nlshrink is R, skfolio and pyrmt are not in the oracle's environment), so the reference is a matrix-form transcription
of the published QIS.m of Ledoit and Wolf (2022, "Quadratic shrinkage for large covariance matrices", Bernoulli 28),
with numpy `eigh` and broadcasting where the TypeScript loops: it guards the implementation against transcription
errors, not the published formula against itself. The statistical behaviour (a lower Frobenius loss than the sample
covariance and the Ledoit-Wolf shrinkage on spiked data) is tested in TypeScript."""

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-12}

type Matrix = npt.NDArray[np.float64]


def qis(y: Matrix, demean: bool = True) -> tuple[Matrix, Matrix, Matrix]:
    """QIS.m: the shrunk covariance, the sample eigenvalues and the shrunk eigenvalues, both descending."""
    big_n, p = y.shape
    if demean:
        y = y - y.mean(axis=0)
        k = 1
    else:
        k = 0
    n = big_n - k
    c = p / n
    sample = y.T @ y / n
    sample = (sample + sample.T) / 2
    lam, u = np.linalg.eigh(sample)
    order = np.argsort(lam)
    lam, u = lam[order], u[:, order]
    kept = lam[max(0, p - n) :]
    m = kept.shape[0]
    h = min(c**2, 1 / c**2) ** 0.35 / p**0.35
    inv = 1.0 / kept
    lj = np.tile(inv[:, None], (1, m))  # lj[i, j] = 1 / lambda_i
    lj_i = lj - lj.T  # (1 / lambda_i) - (1 / lambda_j)
    theta = np.mean(lj * lj_i / (lj_i**2 + h**2 * lj**2), axis=0)
    htheta = np.mean(lj * (h * lj) / (lj_i**2 + h**2 * lj**2), axis=0)
    atheta2 = theta**2 + htheta**2
    if p <= n:
        delta = 1.0 / ((1 - c) ** 2 * inv + 2 * c * (1 - c) * inv * theta + c**2 * inv * atheta2)
    else:
        delta0 = 1.0 / ((c - 1) * np.mean(inv))
        delta = np.concatenate([np.full(p - n, delta0), 1.0 / (inv * atheta2)])
    delta = delta * (np.sum(lam) / np.sum(delta))
    cov = u @ np.diag(delta) @ u.T
    return cov, lam[::-1].copy(), delta[::-1].copy()


def sample_data(seed: int, n: int, p: int) -> Matrix:
    g = rng(seed)
    load = g.standard_normal((p, 3)) * 2.5
    return np.asarray(
        g.standard_normal((n, 3)) @ load.T + g.standard_normal((n, p)) + g.uniform(-1, 1, p), dtype=np.float64
    )


def cases() -> list[Case]:
    out: list[Case] = []
    # A square regime, a wide-sample one, one with more variables than observations, and a near-square one.
    for name, (seed, n, p) in {
        "tall": (3, 200, 40),
        "mid": (4, 80, 60),
        "wide": (5, 40, 60),
        "square": (6, 50, 49),
    }.items():
        x = sample_data(seed, n, p)
        cov, lam, delta = qis(x)
        out.append(case(f"{name}/covariance", "covariance", [x], cov, **TOL))
        out.append(case(f"{name}/sampleEigenvalues", "sampleEigenvalues", [x], lam, **TOL))
        out.append(case(f"{name}/shrunkEigenvalues", "shrunkEigenvalues", [x], delta, **TOL))
    centred = sample_data(7, 60, 20)
    centred = centred - centred.mean(axis=0)
    cov, _, delta = qis(centred, demean=False)
    out.append(case("centred/covariance", "covariance", [centred], cov, **TOL, assumeCentered=True))
    out.append(case("centred/shrunkEigenvalues", "shrunkEigenvalues", [centred], delta, **TOL, assumeCentered=True))
    return out
