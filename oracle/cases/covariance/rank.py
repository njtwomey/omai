"""The nonparanormal correlation estimators and transform against scipy: Spearman and Kendall sine transforms and the
normal scores of the Winsorised empirical distribution (average ranks for ties)."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.special import ndtri

REFERENCE: dict[str, str] = {"scipy": version("scipy"), "numpy": np.__version__}

EXACT = {"rtol": 1e-12, "atol": 1e-12}
SPECIAL = {"rtol": 1e-10, "atol": 1e-10}


def spearman_sine(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """2 sin(pi rho / 6) of the Spearman matrix, with a unit diagonal."""
    rho = np.asarray(st.spearmanr(x).statistic)
    out = 2.0 * np.sin(np.pi * rho / 6.0)
    np.fill_diagonal(out, 1.0)
    return out


def kendall_sine(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """sin(pi tau / 2) of the tau-b matrix, with a unit diagonal."""
    p = x.shape[1]
    out = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            tau = float(st.kendalltau(x[:, i], x[:, j]).statistic)
            out[i, j] = out[j, i] = np.sin(np.pi * tau / 2.0)
    return out


def normal_scores(x: npt.NDArray[np.float64], winsorise: bool) -> npt.NDArray[np.float64]:
    """Standardised normal scores of the (Winsorised) empirical distribution, average ranks."""
    n = x.shape[0]
    delta = 1.0 / (4.0 * n**0.25 * np.sqrt(np.pi * np.log(n)))
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        r = st.rankdata(x[:, j])
        probs = np.clip(r / n, delta, 1 - delta) if winsorise else r / (n + 1)
        z = ndtri(probs)
        out[:, j] = (z - z.mean()) / z.std()
    return out


def cases() -> list[Case]:
    g = rng(31)
    cov = np.array([[1.0, 0.6, 0.2, 0.0], [0.6, 1.0, 0.5, 0.1], [0.2, 0.5, 1.0, 0.4], [0.0, 0.1, 0.4, 1.0]])
    z = g.multivariate_normal(np.zeros(4), cov, 90)
    heavy = np.column_stack([z[:, 0] ** 3, np.exp(z[:, 1]), z[:, 2], np.sinh(2 * z[:, 3])])
    tied = np.round(heavy * 2.0) / 2.0
    out: list[Case] = []
    for name, x in [("heavy", heavy), ("tied", tied)]:
        out.append(case(f"{name}/spearman", "spearman", [x], spearman_sine(x), **EXACT))
        out.append(case(f"{name}/kendall", "kendall", [x], kendall_sine(x), **EXACT))
        out.append(case(f"{name}/transform", "transform", [x], normal_scores(x, True), **SPECIAL, winsorise=True))
        out.append(case(f"{name}/shrunk", "transform", [x], normal_scores(x, False), **SPECIAL, winsorise=False))
    return out
