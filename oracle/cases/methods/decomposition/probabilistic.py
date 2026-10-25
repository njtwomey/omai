"""Factor analysis against scikit-learn: model covariance, noise variances, log likelihood and rotated loadings."""

from importlib.metadata import version
from typing import Literal

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.decomposition import PCA, FactorAnalysis

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}


def data() -> FloatArray:
    g = rng(31)
    z = g.standard_normal((60, 2))
    load = np.array([[1.0, 0.8, 0.6, 0.0, 0.1], [0.0, 0.3, 0.5, 1.0, 0.9]])
    return z @ load + g.standard_normal((60, 5)) * np.array([0.3, 0.5, 0.2, 0.4, 0.6])


def cases() -> list[Case]:
    out: list[Case] = []
    x = data()
    fa = FactorAnalysis(n_components=2).fit(x)
    out.append(case("fa/covariance", "covariance", [x], np.asarray(fa.get_covariance()), **TOL, k=2))
    out.append(case("fa/noise", "noise", [x], np.asarray(fa.noise_variance_), **TOL, k=2))
    out.append(case("fa/score", "score", [x], np.float64(fa.score(x)), **TOL, k=2))
    rotations: list[Literal["varimax", "quartimax"]] = ["varimax", "quartimax"]
    for rotation in rotations:
        r = FactorAnalysis(n_components=2, rotation=rotation).fit(x)
        out.append(
            case(f"fa/{rotation}", "rotated", [x], np.abs(np.asarray(r.components_)), **TOL, k=2, rotation=rotation)
        )
    p = PCA(n_components=2).fit(x)
    out.append(case("ppca/score", "ppcaScore", [x], np.float64(p.score(x)), **TOL, k=2))
    return out
