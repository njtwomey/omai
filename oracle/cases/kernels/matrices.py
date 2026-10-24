"""Kernel-matrix tools against scikit-learn: KernelCenterer (fit, transform of a cross-kernel), KernelPCA on a
precomputed kernel (eigenvalues, sign-fixed eigenvectors, projections, transform of new points) and pairwise_kernels
for every metric with its default parameters.

Kernel alignment, HSIC (biased and unbiased), MMD^2 (biased, unbiased, linear) and the effective dimension have no
function in the oracle libraries; they are formed with numpy from their definitions (Cristianini et al. 2001; Cortes et
al. 2012; Gretton et al. 2005 and 2012; Song et al. 2012; Zhang 2005) on rbf_kernel Gram matrices."""

import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.preprocessing import KernelCenterer
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, sign_fixed, uniform

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def centred(k: FloatArray) -> FloatArray:
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def alignment(a: FloatArray, b: FloatArray) -> float:
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def hsic_biased(k: FloatArray, kl: FloatArray) -> float:
    n = k.shape[0]
    return float(np.trace(centred(k) @ centred(kl)) / (n - 1) ** 2)


def hsic_unbiased(k: FloatArray, kl: FloatArray) -> float:
    n = k.shape[0]
    kt = k - np.diag(np.diag(k))
    lt = kl - np.diag(np.diag(kl))
    one = np.ones(n)
    first = np.trace(kt @ lt)
    second = (one @ kt @ one) * (one @ lt @ one) / ((n - 1) * (n - 2))
    third = 2 / (n - 2) * (one @ kt @ lt @ one)
    return float((first + second - third) / (n * (n - 3)))


def mmd(kxx: FloatArray, kyy: FloatArray, kxy: FloatArray, estimator: str) -> float:
    m, n = kxx.shape[0], kyy.shape[0]
    if estimator == "biased":
        return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())
    if estimator == "unbiased":
        within_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        within_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        return float(within_x + within_y - 2 * kxy.mean())
    pairs = m // 2
    h = [
        kxx[2 * p, 2 * p + 1] + kyy[2 * p, 2 * p + 1] - kxy[2 * p, 2 * p + 1] - kxy[2 * p + 1, 2 * p]
        for p in range(pairs)
    ]
    return float(np.mean(h))


def cases() -> list[Case]:
    g = rng(61)
    x = uniform(g, (10, 3))
    z = uniform(g, (4, 3))
    k = rbf_kernel(x, gamma=0.4)
    kz = rbf_kernel(z, x, gamma=0.4)
    centerer = KernelCenterer().fit(k)
    out = [
        case("center", "center", [k], centerer.transform(k), rtol=1e-12, atol=1e-14),
        case("center-cross", "centerCross", [kz, k], centerer.transform(kz), rtol=1e-12, atol=1e-14),
    ]
    pca = KernelPCA(n_components=4, kernel="precomputed", eigen_solver="dense").fit(k)
    out.append(
        case(
            "kernel-pca",
            "kernelPca",
            [k, kz],
            {
                "eigenvalues": pca.eigenvalues_,
                "eigenvectors": sign_fixed(np.asarray(pca.eigenvectors_, dtype=np.float64)),
                "projections": pca.transform(k),
                "transform": pca.transform(kz),
            },
            rtol=1e-9,
            atol=1e-11,
            nComponents=4,
        )
    )
    y = uniform(g, (10, 2))
    kl = rbf_kernel(y, gamma=1.3)
    lin = x @ x.T
    targets = np.sign(uniform(g, (10,)))
    out.append(
        case(
            "alignment",
            "alignment",
            [k, kl, lin, targets],
            {
                "plain": np.float64(alignment(k, kl)),
                "centred": np.float64(alignment(centred(k), centred(kl))),
                "linear": np.float64(alignment(k, lin)),
                "target": np.float64(alignment(k, np.outer(targets, targets))),
                "targetCentred": np.float64(alignment(centred(k), centred(np.outer(targets, targets)))),
            },
            rtol=1e-12,
        )
    )
    # A dependent pair: y depends on x through its first coordinate.
    dep = np.stack([np.sin(2 * x[:, 0]), x[:, 0] ** 2], axis=1) + 0.05 * uniform(g, (10, 2))
    ld = rbf_kernel(dep, gamma=1.0)
    out.append(
        case(
            "hsic",
            "hsic",
            [k, kl, ld],
            {
                "biased": np.float64(hsic_biased(k, kl)),
                "unbiased": np.float64(hsic_unbiased(k, kl)),
                "biasedDependent": np.float64(hsic_biased(k, ld)),
                "unbiasedDependent": np.float64(hsic_unbiased(k, ld)),
                "normalised": np.float64(hsic_biased(k, ld) / np.sqrt(hsic_biased(k, k) * hsic_biased(ld, ld))),
            },
            rtol=1e-11,
            atol=1e-14,
        )
    )
    xs = uniform(g, (8, 2))
    ys = uniform(g, (8, 2), -1.0, 3.0)
    kxx = rbf_kernel(xs, gamma=0.5)
    kyy = rbf_kernel(ys, gamma=0.5)
    kxy = rbf_kernel(xs, ys, gamma=0.5)
    out.append(
        case(
            "mmd",
            "mmd",
            [kxx, kyy, kxy],
            {e: np.float64(mmd(kxx, kyy, kxy, e)) for e in ["biased", "unbiased", "linear"]},
            rtol=1e-12,
            atol=1e-14,
        )
    )
    for lam in [0.01, 1.0]:
        eff = np.trace(k @ np.linalg.inv(k + lam * np.eye(k.shape[0])))
        out.append(case(f"effective-dimension-{lam}", "effectiveDimension", [k], np.float64(eff), rtol=1e-10, lam=lam))
    a = uniform(g, (6, 3), 0.0, 2.0)
    b = uniform(g, (4, 3), 0.0, 2.0)
    for metric in ["additive_chi2", "chi2", "linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"]:
        out.append(case(f"pairwise-{metric}", "pairwise", [a, b], pairwise_kernels(a, b, metric=metric), metric=metric))
        out.append(
            case(f"pairwise-{metric}-gram", "pairwiseGram", [a], pairwise_kernels(a, metric=metric), metric=metric)
        )
    out.append(
        case(
            "pairwise-rbf-gamma",
            "pairwise",
            [a, b],
            pairwise_kernels(a, b, metric="rbf", gamma=2.5),
            metric="rbf",
            gamma=2.5,
        )
    )
    return out
