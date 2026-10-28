"""Sparse Gaussian-process regression against dense numpy forms of the approximations: the marginal likelihood
log N(y | 0, Q + Lambda) (less the trace term for the variational bound) and the predictive mean and variance, formed
with the n x n matrices the engine avoids."""

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.spatial.distance import cdist

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

JITTER = 1e-6


def dense(
    approximation: str, x: FloatArray, y: FloatArray, z: FloatArray, xt: FloatArray, length: float, sigma2: float
) -> dict[str, FloatArray]:

    def kernel(a: FloatArray, b: FloatArray | None = None) -> FloatArray:
        return np.asarray(np.exp(-0.5 * cdist(a, a if b is None else b, "sqeuclidean") / length**2), dtype=np.float64)

    n = x.shape[0]
    kmm = kernel(z)
    kmm = kmm + JITTER * np.mean(np.diag(kmm)) * np.eye(z.shape[0])
    knm = kernel(x, z)
    q = knm @ np.linalg.solve(kmm, knm.T)
    diag_k = np.ones(n)
    lam = sigma2 + np.maximum(diag_k - np.diag(q), 0.0) if approximation == "fitc" else sigma2 * np.ones(n)
    cov = q + np.diag(lam)
    _, logdet = np.linalg.slogdet(cov)
    ll = -0.5 * y @ np.linalg.solve(cov, y) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    trace = (np.sum(diag_k) - np.trace(q)) / (2 * sigma2) if approximation == "vfe" else 0.0
    # Predictive: Sigma_B = (Kmm + Kmn Lambda^-1 Knm)^-1.
    sigma_b = np.linalg.inv(kmm + knm.T @ (knm / lam[:, None]))
    kts = kernel(xt, z)
    mean = kts @ sigma_b @ knm.T @ (y / lam)
    explained = np.einsum("ij,jk,ik->i", kts, sigma_b, kts)
    projected = np.einsum("ij,ij->i", kts, np.linalg.solve(kmm, kts.T).T)
    variance = explained if approximation == "sor" else 1.0 - projected + explained
    return {"bound": np.array(ll - trace), "mean": mean, "variance": variance}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(71)
    x = np.sort(g.uniform(-3.0, 3.0, size=(30, 1)), axis=0)
    y = np.sin(1.4 * x[:, 0]) + 0.1 * g.normal(size=30)
    xt = np.linspace(-3.5, 3.5, 9).reshape(-1, 1)
    z = np.linspace(-2.8, 2.8, 6).reshape(-1, 1)
    for approximation in ("sor", "dtc", "fitc", "vfe"):
        for name, inducing in [("m6", z), ("all", x)]:
            output = dense(approximation, x, y, inducing, xt, 1.1, 0.04)
            out.append(
                case(
                    f"{approximation}/{name}",
                    "sparseGp",
                    [x, y, xt, inducing],
                    output,
                    # With all 30 points inducing, the inducing kernel matrix is ill conditioned and the two
                    # evaluation orders differ in the last digits.
                    rtol=1e-6 if name == "m6" else 1e-4,
                    atol=1e-8 if name == "m6" else 1e-6,
                    approximation=approximation,
                    lengthScale=1.1,
                    noise=0.04,
                )
            )
    return out
