"""Gaussian processes: regression posteriors and log marginal likelihoods against
sklearn.gaussian_process.GaussianProcessRegressor with fixed kernels (optimizer=None), the covariance matrices of
Brownian motion, the Brownian bridge and the stationary Ornstein-Uhlenbeck process, and the Nystrom Karhunen-Loeve
eigenpairs (numpy.linalg.eigh of the trapezoid-weighted covariance)."""

from typing import cast

import numpy as np
import scipy
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def first_positive(v: FloatArray) -> FloatArray:
    """Flip each column so that its first clearly non-zero entry is positive (the eigenfunction starts increasing)."""
    out = v.copy()
    for c in range(v.shape[1]):
        col = v[:, c]
        first = col[np.abs(col) > 1e-6 * np.abs(col).max()][0]
        out[:, c] = col * np.sign(first)
    return out


def cases() -> list[Case]:
    g = rng(77)
    out: list[Case] = []
    x = np.sort(g.uniform(0, 5, size=8))
    y = np.sin(x) + 0.1 * g.normal(size=8)
    xs = np.linspace(-0.5, 5.5, 6)
    for variance, length, noise in [(1.0, 1.0, 1e-2), (2.0, 0.5, 0.1)]:
        kernel = ConstantKernel(variance, "fixed") * RBF(length, "fixed")
        # optimizer=None keeps the kernel fixed (the stubs type the argument as a string only)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, optimizer=cast(str, None)).fit(x[:, None], y)
        mean, cov = gp.predict(xs[:, None], return_cov=True)
        _, std = gp.predict(xs[:, None], return_std=True)
        params = {"variance": variance, "length": length, "noise": noise}
        tag = f"{variance}/{length}"
        inputs = [x, y, xs]
        out.append(case(f"gp/mean/{tag}", "gp-mean", inputs, mean, rtol=1e-9, atol=1e-12, **params))
        out.append(case(f"gp/cov/{tag}", "gp-cov", inputs, cov, rtol=1e-8, atol=1e-11, **params))
        out.append(case(f"gp/std/{tag}", "gp-std", inputs, std, rtol=1e-8, atol=1e-10, **params))
        lml = np.array(gp.log_marginal_likelihood_value_)
        out.append(case(f"gp/lml/{tag}", "gp-lml", inputs, lml, rtol=1e-9, **params))
    t = np.array([0.1, 0.4, 0.5, 0.9])
    out.append(case("cov/brownian", "cov-brownian", [t], 2.0 * np.minimum.outer(t, t), sigma=np.sqrt(2.0)))
    out.append(
        case("cov/bridge", "cov-bridge", [t], np.minimum.outer(t, t) - np.outer(t, t) / 1.0, horizon=1.0, sigma=1.0)
    )
    theta, sigma = 1.5, 0.8
    ou = sigma**2 / (2 * theta) * np.exp(-theta * np.abs(np.subtract.outer(t, t)))
    out.append(case("cov/ou", "cov-ou", [t], ou, rtol=1e-12, theta=theta, sigma=sigma))
    grid = np.linspace(0, 1, 41)
    w = np.full(41, 1 / 40)
    w[0] = w[-1] = 1 / 80
    root = np.sqrt(w)
    k = np.minimum.outer(grid, grid)
    vals, vecs = np.linalg.eigh(root[:, None] * k * root[None, :])
    top = np.argsort(vals)[::-1][:5]
    funcs = first_positive(vecs[:, top] / root[:, None])
    out.append(case("kl/values", "kl-values", [grid], vals[top], rtol=1e-9, atol=1e-13, terms=5))
    out.append(case("kl/functions", "kl-functions", [grid], funcs.T, rtol=1e-7, atol=1e-9, terms=5))
    return out
