"""Stationary kernels against scikit-learn (RBF, Matern for nu in {0.5, 1.5, 2.5, inf} and general orders through
scipy's kv, RationalQuadratic, ExpSineSquared, ExpSineSquared * RBF, laplacian_kernel) with their eval_gradient
gradients, GPyTorch (SpectralMixtureKernel in one dimension, where its sum-then-product order agrees with Wilson and
Adams, and CosineKernel), and the gamma-exponential kernel formed from scipy's cdist.

scikit-learn differentiates Matern of a general order numerically (`_approx_fprime`, step 1e-10), so those gradients
are compared at 1e-5 (the reference's own accuracy)."""

import gpytorch
import numpy as np
import scipy
import sklearn
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, uniform
from omai_oracle.kernels import anisotropic, gp_case
from omai_oracle.torch_helpers import to_numpy, to_torch
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, RationalQuadratic
from sklearn.metrics.pairwise import laplacian_kernel

REFERENCE: dict[str, str] = {
    "scikit-learn": sklearn.__version__,
    "gpytorch": gpytorch.__version__,
    "scipy": scipy.__version__,
}


def spectral_mixture(x: FloatArray, y: FloatArray, w: FloatArray, mu: FloatArray, s: FloatArray) -> FloatArray:
    """GPyTorch SpectralMixtureKernel with weights [Q], means [Q, 1] and scales [Q, 1], on one-dimensional inputs."""
    k = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=len(w), ard_num_dims=1).double()
    k.mixture_weights = torch.tensor(w, dtype=torch.float64)
    k.mixture_means = torch.tensor(mu.reshape(-1, 1, 1), dtype=torch.float64)
    k.mixture_scales = torch.tensor(s.reshape(-1, 1, 1), dtype=torch.float64)
    with torch.no_grad():
        return to_numpy(k(to_torch(x), to_torch(y)).to_dense())


def cosine(x: FloatArray, y: FloatArray, period: float) -> FloatArray:
    """GPyTorch CosineKernel with a period."""
    k = gpytorch.kernels.CosineKernel().double()
    k.period_length = torch.tensor(period, dtype=torch.float64)
    with torch.no_grad():
        return to_numpy(k(to_torch(x), to_torch(y)).to_dense())


def cases() -> list[Case]:
    g = rng(21)
    x = uniform(g, (8, 2))
    y = uniform(g, (5, 2))
    x1 = uniform(g, (7, 1), -3, 3)
    y1 = uniform(g, (4, 1), -3, 3)
    out = [
        gp_case("rbf", RBF(1.3), x, y, kernel="rbf", lengthScale=1.3),
        gp_case("rbf-ard", RBF(anisotropic([0.5, 2.0])), x, y, kernel="rbf", lengthScale=[0.5, 2.0]),
        gp_case("rational-quadratic", RationalQuadratic(0.9, 1.7), x, y, rtol=1e-11, kernel="rq"),
        gp_case("exp-sine-squared", ExpSineSquared(1.2, 2.5), x, y, rtol=1e-10, kernel="periodic"),
        gp_case("exp-sine-squared-1d", ExpSineSquared(0.7, 1.3), x1, y1, rtol=1e-10, kernel="periodic-1d"),
        gp_case(
            "exp-sine-squared*rbf",
            ExpSineSquared(1.1, 3.0) * RBF(2.0),
            x,
            y,
            rtol=1e-10,
            kernel="locally-periodic",
        ),
    ]
    for nu in [0.5, 1.5, 2.5, float("inf")]:
        out.append(gp_case(f"matern-{nu}", Matern(0.8, nu=nu), x, y, rtol=1e-11, kernel="matern", nu=str(nu)))
    out.append(
        gp_case(
            "matern-1.5-ard", Matern(anisotropic([0.6, 1.7]), nu=1.5), x, y, rtol=1e-11, kernel="matern-ard", nu="1.5"
        )
    )
    for nu in [0.7, 3.2]:
        # Values through kv at 1e-10; the reference gradient is a forward difference (see the module docstring).
        out.append(
            gp_case(f"matern-{nu}", Matern(1.1, nu=nu), x, y, rtol=1e-5, atol=1e-7, kernel="matern-general", nu=str(nu))
        )
        out.append(
            case(
                f"matern-{nu}/values",
                "matern",
                [x, y],
                np.asarray(Matern(1.1, nu=nu)(x, y), dtype=np.float64),
                rtol=1e-10,
                atol=1e-14,
                nu=nu,
                lengthScale=1.1,
            )
        )
    out.append(case("laplacian", "laplacian", [x, y], laplacian_kernel(x, y, gamma=0.7), rtol=1e-12, gamma=0.7))
    for power in [1.0, 1.5]:
        ell = np.array([0.9, 1.6])
        ref = np.exp(-(cdist(x / ell, y / ell) ** power))
        out.append(case(f"gamma-exponential-{power}", "exponential", [x, y], ref, rtol=1e-12, power=power))
    w = np.array([0.6, 1.3, 0.4])
    mu = np.array([0.15, 0.8, 1.6])
    s = np.array([0.3, 0.12, 0.5])
    out.append(
        case(
            "spectral-mixture",
            "spectralMixture",
            [x1, y1, w, mu.reshape(-1, 1), s.reshape(-1, 1)],
            spectral_mixture(x1, y1, w, mu, s),
            rtol=1e-12,
        )
    )
    out.append(case("cosine", "cosine", [x1, y1], cosine(x1, y1, 1.7), rtol=1e-12, period=1.7))
    return out
