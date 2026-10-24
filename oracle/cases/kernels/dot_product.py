"""Dot-product kernels against scikit-learn (DotProduct with eval_gradient, linear_kernel, polynomial_kernel,
sigmoid_kernel, cosine_similarity) and GPyTorch (LinearKernel, PolynomialKernel).

The arc-cosine kernel (Cho and Saul 2009) and the infinite-width NTK and NNGP kernels have no reference in the oracle
groups: neural-tangents, their usual reference, does not import with the installed jax (it needs the removed
`jax.linear_util`). They are formed with numpy from the published closed forms: J_n(theta) for the arc-cosine kernel,
and for the NTK the recursion Sigma <- sigma_w^2 E[phi phi] + sigma_b^2, Theta <- sigma_w^2 Theta E[phi' phi'] + Sigma
(neural-tangents' Dense/Relu/Erf layers in the NTK parameterisation, inputs divided by their dimension). The TypeScript
tests also check both against Monte Carlo and against the empirical NTK of wide random networks."""

import gpytorch
import numpy as np
import sklearn
import torch
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, polynomial_kernel, sigmoid_kernel
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, uniform
from whitebox_oracle.kernels import call, gp_case
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "gpytorch": gpytorch.__version__}


def gpytorch_linear(x: FloatArray, y: FloatArray, variance: float) -> FloatArray:
    """GPyTorch LinearKernel(variance)."""
    k = gpytorch.kernels.LinearKernel().double()
    k.variance = torch.tensor(variance, dtype=torch.float64)
    with torch.no_grad():
        return to_numpy(k(to_torch(x), to_torch(y)).to_dense())


def gpytorch_polynomial(x: FloatArray, y: FloatArray, power: int, offset: float) -> FloatArray:
    """GPyTorch PolynomialKernel(power, offset): (x . y + offset)^power."""
    k = gpytorch.kernels.PolynomialKernel(power=power).double()
    k.offset = torch.tensor(offset, dtype=torch.float64)
    with torch.no_grad():
        return to_numpy(k(to_torch(x), to_torch(y)).to_dense())


def cosines(sxy: FloatArray, sxx: FloatArray, syy: FloatArray, gram: bool) -> FloatArray:
    """Cosines of the angles, clipped to [-1, 1], with an exact unit diagonal for a Gram matrix."""
    c = np.clip(sxy / np.sqrt(np.outer(sxx, syy)), -1.0, 1.0)
    if gram:
        np.fill_diagonal(c, 1.0)
    return c


def j(order: int, theta: FloatArray) -> FloatArray:
    """Cho and Saul's angular profile J_n(theta)."""
    if order == 0:
        return np.pi - theta
    if order == 1:
        return np.sin(theta) + (np.pi - theta) * np.cos(theta)
    return 3 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (1 + 2 * np.cos(theta) ** 2)


def arc_cosine(x: FloatArray, y: FloatArray | None, order: int, depth: int) -> FloatArray:
    """The arc-cosine kernel of the given order composed `depth` times."""
    other = x if y is None else y
    k = x @ other.T
    dx = (x**2).sum(1)
    dy = (other**2).sum(1)
    self_factor = [1.0, 1.0, 3.0][order]
    for _ in range(depth):
        theta = np.arccos(cosines(k, dx, dy, y is None))
        k = np.outer(dx, dy) ** (order / 2) * j(order, theta) / np.pi
        dx = np.ones_like(dx) if order == 0 else dx**order * self_factor
        dy = np.ones_like(dy) if order == 0 else dy**order * self_factor
    return k


def ntk(
    x: FloatArray, y: FloatArray | None, depth: int, wv: float, bv: float, activation: str
) -> tuple[FloatArray, FloatArray]:
    """The infinite-width NTK and NNGP kernels of Dense -> (activation -> Dense) x depth."""
    other = x if y is None else y
    d = x.shape[1]
    sxy = wv * (x @ other.T) / d + bv
    sxx = wv * (x**2).sum(1) / d + bv
    syy = wv * (other**2).sum(1) / d + bv
    theta_k = sxy.copy()
    for _ in range(depth):
        if activation == "relu":
            c = cosines(sxy, sxx, syy, y is None)
            t = np.arccos(c)
            value = np.sqrt(np.outer(sxx, syy)) * j(1, t) / (2 * np.pi)
            derivative = (np.pi - t) / (2 * np.pi)
            sxx_next, syy_next = sxx / 2, syy / 2
        else:
            a = 1 + 2 * sxx
            b = 1 + 2 * syy
            ab = np.outer(a, b)
            value = 2 / np.pi * np.arcsin(2 * sxy / np.sqrt(ab))
            derivative = 4 / np.pi / np.sqrt(ab - 4 * sxy**2)
            sxx_next = 2 / np.pi * np.arcsin(2 * sxx / a)
            syy_next = 2 / np.pi * np.arcsin(2 * syy / b)
        sxy = wv * value + bv
        sxx = wv * sxx_next + bv
        syy = wv * syy_next + bv
        theta_k = wv * theta_k * derivative + sxy
    return theta_k, sxy


def cases() -> list[Case]:
    g = rng(31)
    x = uniform(g, (8, 3))
    y = uniform(g, (5, 3))
    yz = y.copy()
    yz[2] = 0.0
    out = [
        gp_case("dot-product", DotProduct(0.8), x, y, kernel="dot-product"),
        case("linear", "linear", [x, y], linear_kernel(x, y)),
        case("gpytorch-linear", "linearVariance", [x, y], gpytorch_linear(x, y, 1.6), variance=1.6),
        case(
            "polynomial",
            "polynomial",
            [x, y],
            call(polynomial_kernel, x, y, degree=3, gamma=0.4, coef0=1.2),
            degree=3,
            gamma=0.4,
            coef0=1.2,
        ),
        case(
            "gpytorch-polynomial",
            "polynomial",
            [x, y],
            gpytorch_polynomial(x, y, 2, 0.5),
            degree=2,
            gamma=1.0,
            coef0=0.5,
        ),
        case("sigmoid", "sigmoid", [x, y], call(sigmoid_kernel, x, y, gamma=0.3, coef0=-0.2), gamma=0.3, coef0=-0.2),
        case("cosine-similarity", "cosineSimilarity", [x, yz], cosine_similarity(x, yz)),
        case("cosine-similarity-gram", "cosineSimilarityGram", [x], cosine_similarity(x)),
    ]
    for order in [0, 1, 2]:
        for depth in [1, 2]:
            out.append(
                case(
                    f"arc-cosine-{order}-{depth}",
                    "arcCosine",
                    [x, y],
                    {"cross": arc_cosine(x, y, order, depth), "gram": arc_cosine(x, None, order, depth)},
                    rtol=1e-12,
                    order=order,
                    depth=depth,
                )
            )
    for activation, depth, wv, bv in [("relu", 1, 2.0, 0.0), ("relu", 3, 1.5, 0.1), ("erf", 2, 1.2, 0.3)]:
        cross_ntk, cross_nngp = ntk(x, y, depth, wv, bv, activation)
        gram_ntk, gram_nngp = ntk(x, None, depth, wv, bv, activation)
        out.append(
            case(
                f"ntk-{activation}-{depth}",
                "ntk",
                [x, y],
                {"ntk": cross_ntk, "nngp": cross_nngp, "ntkGram": gram_ntk, "nngpGram": gram_nngp},
                rtol=1e-12,
                activation=activation,
                depth=depth,
                weightVariance=wv,
                biasVariance=bv,
            )
        )
    return out
