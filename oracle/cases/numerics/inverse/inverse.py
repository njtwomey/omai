"""Regularisation against straightforward numpy implementations (stacked least squares, SVD filter formulas) and
scipy.sparse.linalg.lsqr."""

import numpy as np
import scipy
import scipy.sparse.linalg as sla
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def problem(n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """A Gaussian blurring operator (ill-posed), a smooth true signal and noisy data."""
    t = (np.arange(n) + 0.5) / n
    a = np.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * 0.03**2)) / n
    x = np.sin(2 * np.pi * t) + 0.5 * np.exp(-((t - 0.6) ** 2) / 0.005)
    b = a @ x + 1e-3 * rng(130).normal(size=n)
    return a, x, b


def cases() -> list[Case]:
    a, _, b = problem(40)
    out: list[Case] = []
    u, s, vh = np.linalg.svd(a)
    beta = u.T @ b
    for lam in (1e-4, 1e-3, 1e-2):
        stacked = np.vstack([a, lam * np.eye(40)])
        x = np.linalg.lstsq(stacked, np.concatenate([b, np.zeros(40)]), rcond=None)[0]
        out.append(case(f"tikhonov/{lam}", "tikhonov", [a, b], x, rtol=1e-6, atol=1e-8, lam=lam))
        out.append(case(f"general/{lam}", "general", [a, b], x, rtol=1e-6, atol=1e-8, lam=lam))
    l2 = np.diff(np.eye(40), 2, axis=0)
    for lam in (1e-3, 1e-2):
        stacked = np.vstack([a, lam * l2])
        x = np.linalg.lstsq(stacked, np.concatenate([b, np.zeros(38)]), rcond=None)[0]
        out.append(case(f"general/second/{lam}", "generalSecond", [a, b], x, rtol=1e-6, atol=1e-8, lam=lam))
    for k in (5, 10):
        x = vh[:k].T @ (beta[:k] / s[:k])
        out.append(case(f"tsvd/{k}", "tsvd", [a, b], x, rtol=1e-6, atol=1e-8, k=k))
    lams = np.logspace(-6, 0, 25)
    g = np.array(
        [np.sum(((lam**2 / (s**2 + lam**2)) * beta) ** 2) / (40 - np.sum(s**2 / (s**2 + lam**2))) ** 2 for lam in lams]
    )
    out.append(case("gcv/function", "gcvFunction", [a, b, lams], g, rtol=1e-6, atol=1e-14))
    res = sla.lsqr(a, b, iter_lim=8, atol=0, btol=0, conlim=0)
    out.append(case("lsqr/8", "lsqr", [a, b], np.asarray(res[0], dtype=np.float64), rtol=1e-7, atol=1e-9, iterations=8))
    return out
