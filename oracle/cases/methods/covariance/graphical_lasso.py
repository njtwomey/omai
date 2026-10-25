"""The graphical lasso against scikit-learn: precision and covariance at two penalties, tolerances tightened."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.covariance import empirical_covariance, graphical_lasso

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-6, "atol": 1e-8}


def cases() -> list[Case]:
    g = rng(23)
    prec = np.array([[2.0, 0.6, 0, 0], [0.6, 2.0, 0.5, 0], [0, 0.5, 2.0, 0.4], [0, 0, 0.4, 2.0]])
    x = g.multivariate_normal(np.zeros(4), np.linalg.inv(prec), 80)
    s = empirical_covariance(x)
    out: list[Case] = []
    for alpha in [0.02, 0.1]:
        cov, pre = graphical_lasso(s, alpha, tol=1e-10, enet_tol=1e-12, max_iter=500)
        out.append(case(f"alpha{alpha}/precision", "precision", [s], np.asarray(pre), **TOL, alpha=alpha))
        out.append(case(f"alpha{alpha}/covariance", "covariance", [s], np.asarray(cov), **TOL, alpha=alpha))
    return out
