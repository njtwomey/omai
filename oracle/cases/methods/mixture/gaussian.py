"""Gaussian mixtures by EM from fixed initial parameters against scikit-learn: weights, means, covariances, scores."""

from importlib.metadata import version
from typing import Any, Literal

import numpy as np
from sklearn.mixture import GaussianMixture
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}

Kind = Literal["full", "tied", "diag", "spherical"]


def data() -> FloatArray:
    g = rng(5)
    a = g.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 30)
    b = g.multivariate_normal([4, 1], [[0.5, 0], [0, 2]], 30)
    c = g.multivariate_normal([1, 5], [[1.5, -0.4], [-0.4, 0.7]], 30)
    return np.concatenate([a, b, c])


def initial(kind: Kind) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Weights, means, the full covariance matrices (for the engine) and sklearn's precisions_init of that type."""
    weights = np.array([0.3, 0.3, 0.4])
    means = np.array([[0.5, 0.5], [3.0, 0.0], [0.0, 4.0]])
    full = np.array([[[1.0, 0.2], [0.2, 1.0]], [[1.2, 0.0], [0.0, 0.8]], [[0.9, -0.1], [-0.1, 1.1]]])
    if kind == "tied":
        full = np.repeat(full[:1], 3, axis=0)
        precisions = np.linalg.inv(full[0])
    elif kind == "diag":
        full = np.array([np.diag(np.diag(c)) for c in full])
        precisions = 1 / np.array([np.diag(c) for c in full])
    elif kind == "spherical":
        v = np.array([1.0, 0.9, 1.2])
        full = np.array([s * np.eye(2) for s in v])
        precisions = 1 / v
    else:
        precisions = np.linalg.inv(full)
    return weights, means, full, precisions


def cases() -> list[Case]:
    out: list[Case] = []
    x = data()
    kinds: list[Kind] = ["full", "tied", "diag", "spherical"]
    for kind in kinds:
        w, mu, full, prec = initial(kind)
        options: dict[str, Any] = {
            "n_components": 3,
            "covariance_type": kind,
            "weights_init": w,
            "means_init": mu,
            "precisions_init": prec,
            "tol": 1e-12,
            "max_iter": 15,
        }
        gm = GaussianMixture(**options).fit(x)
        inputs = [x, w, mu, full]
        out.append(case(f"{kind}/weights", "weights", inputs, np.asarray(gm.weights_), **TOL, kind=kind))
        out.append(case(f"{kind}/means", "means", inputs, np.asarray(gm.means_), **TOL, kind=kind))
        out.append(case(f"{kind}/covariances", "covariances", inputs, np.asarray(gm.covariances_), **TOL, kind=kind))
        out.append(case(f"{kind}/score", "score", inputs, np.float64(gm.score(x)), **TOL, kind=kind))
    return out
