"""Kernel algebra against scikit-learn's Gaussian-process kernels (Sum, Product, ConstantKernel * k, Exponentiation,
anisotropic length scales, fixed hyperparameters): Gram matrices, cross-covariances, diagonals and the gradients in
log-hyperparameters (`eval_gradient=True`), with theta for the ordering; and GPyTorch's ScaleKernel over an ARD RBF."""

import gpytorch
import sklearn
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, uniform
from omai_oracle.kernels import anisotropic, gp_case
from omai_oracle.torch_helpers import to_numpy, to_torch
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Exponentiation,
    RationalQuadratic,
    WhiteKernel,
)

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "gpytorch": gpytorch.__version__}


def scale_ard(x: FloatArray, y: FloatArray) -> FloatArray:
    """GPyTorch ScaleKernel(RBFKernel(ard_num_dims=2)) with outputscale 1.7 and length scales [0.6, 1.4]."""
    k = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)).double()
    k.outputscale = torch.tensor(1.7, dtype=torch.float64)
    k.base_kernel.lengthscale = torch.tensor([[0.6, 1.4]], dtype=torch.float64)
    with torch.no_grad():
        return to_numpy(k(to_torch(x), to_torch(y)).to_dense())


def cases() -> list[Case]:
    g = rng(11)
    x = uniform(g, (8, 2))
    y = uniform(g, (5, 2))
    out = [
        gp_case(
            "constant*rbf+white",
            ConstantKernel(2.0) * RBF(1.5) + WhiteKernel(0.1),
            x,
            y,
            kernel="constant*rbf+white",
        ),
        gp_case(
            "rbf-ard*dot-product",
            RBF(anisotropic([0.7, 1.9])) * DotProduct(0.5),
            x,
            y,
            rtol=1e-10,
            kernel="rbf-ard*dot-product",
        ),
        gp_case(
            "rational-quadratic**2",
            Exponentiation(RationalQuadratic(length_scale=1.2, alpha=0.8), 2.0),
            x,
            y,
            rtol=1e-10,
            kernel="rational-quadratic**2",
        ),
        gp_case(
            "fixed-rbf+white",
            RBF(1.0, length_scale_bounds="fixed") + WhiteKernel(0.3),
            x,
            y,
            kernel="fixed-rbf+white",
        ),
        gp_case(
            "(rbf+dot-product)*constant",
            (RBF(0.8) + DotProduct(1.1)) * ConstantKernel(0.6),
            x,
            y,
            rtol=1e-10,
            kernel="(rbf+dot-product)*constant",
        ),
        # GPyTorch forms squared distances as |a|^2 + |b|^2 - 2 a.b (after centring), so 1e-10 rather than 1e-12.
        case("gpytorch-scale-ard", "scaleArd", [x, y], scale_ard(x, y), rtol=1e-10),
    ]
    return out
