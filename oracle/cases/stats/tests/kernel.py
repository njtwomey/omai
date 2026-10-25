"""Kernel tests (MMD and HSIC). Neither numpy nor scipy implements them (hyppo is not an oracle dependency), so the
references are Gretton et al.'s statistics and gamma approximations written in numpy, as in their MATLAB code."""

import numpy as np
import scipy
import scipy.stats as st
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def sqdist(a: FloatArray, b: FloatArray) -> FloatArray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


def median_sigma(z: FloatArray) -> float:
    d = sqdist(z, z)
    iu = np.triu_indices(z.shape[0], 1)
    v = d[iu]
    return float(np.sqrt(0.5 * np.median(v[v > 0])))


def gaussian(a: FloatArray, b: FloatArray, sigma: float) -> FloatArray:
    return np.exp(-sqdist(a, b) / (2 * sigma * sigma))


def mmd_gamma(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    z = np.concatenate([x, y])
    s = median_sigma(z)
    m = x.shape[0]
    K = gaussian(x, x, s)
    L = gaussian(y, y, s)
    KL = gaussian(x, y, s)
    stat = (K.sum() + L.sum() - 2 * KL.sum()) / (m * m)
    mean = 2 / m * (1 - np.trace(KL) / m)
    H = K + L - KL - KL.T
    np.fill_diagonal(H, 0.0)
    var = 2 / m / (m - 1) / m / (m - 1) * (H**2).sum()
    al = mean**2 / var
    bet = var * m / mean
    return float(stat), float(st.gamma.sf(m * stat, al, scale=bet))


def hsic_gamma(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    n = x.shape[0]
    K = gaussian(x, x, median_sigma(x))
    L = gaussian(y, y, median_sigma(y))
    H = np.eye(n) - 1 / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    stat = (Kc * Lc).sum() / (n * n)
    v = (Kc * Lc / 6) ** 2
    var = (v.sum() - np.trace(v)) / n / (n - 1)
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var
    K0 = K - np.diag(np.diag(K))
    L0 = L - np.diag(np.diag(L))
    mux = K0.sum() / n / (n - 1)
    muy = L0.sum() / n / (n - 1)
    mean = (1 + mux * muy - mux - muy) / n
    al = mean**2 / var
    bet = var * n / mean
    return float(stat), float(st.gamma.sf(n * stat, al, scale=bet))


def cases() -> list[Case]:
    g = rng(606)
    x = g.normal(size=(20, 2))
    y = g.normal(size=(20, 2)) * np.array([1.0, 1.6]) + np.array([0.3, 0.0])
    a = g.normal(size=(30, 1))
    b = a**2 + 0.5 * g.normal(size=(30, 1))
    out: list[Case] = [
        case("medianHeuristic", "medianHeuristic", [x, x], np.asarray(median_sigma(x))),
        case("mmdGamma", "mmdGamma", [x, y], np.asarray(mmd_gamma(x, y)), rtol=1e-10),
        case("hsicGamma", "hsicGamma", [a, b], np.asarray(hsic_gamma(a, b)), rtol=1e-10),
    ]
    z = np.concatenate([x, y])
    s = median_sigma(z)
    K = gaussian(x, x, s)
    L = gaussian(y, y, s)
    KL = gaussian(x, y, s)
    m = x.shape[0]
    unbiased = (
        (K.sum() - np.trace(K)) / (m * (m - 1)) + (L.sum() - np.trace(L)) / (m * (m - 1)) - 2 * KL.sum() / (m * m)
    )
    out.append(case("mmd2/unbiased", "mmd2Unbiased", [x, y], np.asarray(unbiased)))
    return out
