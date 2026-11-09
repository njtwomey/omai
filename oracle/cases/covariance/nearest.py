"""Nearest correlation and covariance matrices against statsmodels: `corr_nearest` (Higham's alternating projections
with Dykstra's correction), `corr_clipped` and `cov_nearest` (clipped and nearest), on indefinite matrices of the kind
pairwise and thresholded estimates give."""

from importlib.metadata import version
from warnings import catch_warnings, simplefilter

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.stats.correlation_tools import corr_clipped, corr_nearest, cov_nearest

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

# The projection is unique, but statsmodels stops after `n_fact * p` sweeps: the matrices below converge well inside
# that, which the generator checks against a run with a hundred times the budget.
TOL = {"rtol": 1e-6, "atol": 1e-7}


def pairwise_like(g: np.random.Generator, p: int, noise: float) -> npt.NDArray[np.float64]:
    """A correlation matrix of a low-rank signal with each pair perturbed independently, so it is indefinite."""
    f = g.standard_normal((p, 2))
    c = f @ f.T + np.eye(p)
    d = np.sqrt(np.diag(c))
    corr = c / d[:, None] / d[None, :]
    e = noise * g.standard_normal((p, p))
    out = corr + (e + e.T) / 2
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -1.0, 1.0)


def settled(corr: npt.NDArray[np.float64], result: npt.NDArray[np.float64]) -> bool:
    """Whether statsmodels' default budget reached what a hundred times the budget reaches."""
    with catch_warnings():
        simplefilter("ignore")
        long = corr_nearest(corr, threshold=1e-15, n_fact=10000)
    return bool(np.max(np.abs(long - result)) < 1e-9)


def cases() -> list[Case]:
    g = rng(41)
    higham = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    chain = np.array([[1.0, 0.9, 0.4, 0.0], [0.9, 1.0, 0.9, 0.4], [0.4, 0.9, 1.0, 0.9], [0.0, 0.4, 0.9, 1.0]])
    matrices = {
        "higham3": higham,
        "chain4": chain,
        "pairwise6": pairwise_like(g, 6, 0.35),
        "pairwise8": pairwise_like(g, 8, 0.45),
    }
    out: list[Case] = []
    for name, m in matrices.items():
        with catch_warnings():
            simplefilter("ignore")
            nearest = corr_nearest(m, threshold=1e-15)
        assert settled(m, nearest), name
        out.append(case(f"{name}/corrNearest", "corrNearest", [m], nearest, **TOL))
        out.append(
            case(f"{name}/corrClipped", "corrClipped", [m], corr_clipped(m, threshold=1e-8), **TOL, threshold=1e-8)
        )
        # A covariance: the correlation matrix scaled by standard deviations.
        s = g.uniform(0.5, 3.0, m.shape[0])
        cov = m * s[:, None] * s[None, :]
        for method in ("clipped", "nearest"):
            with catch_warnings():
                simplefilter("ignore")
                result = cov_nearest(cov, method=method, threshold=1e-8 if method == "clipped" else 1e-15)
            out.append(
                case(
                    f"{name}/covNearest/{method}",
                    "covNearest",
                    [cov],
                    np.asarray(result),
                    **TOL,
                    method=method,
                    threshold=1e-8 if method == "clipped" else 1e-15,
                )
            )
    return out
