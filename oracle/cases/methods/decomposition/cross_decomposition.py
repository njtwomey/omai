"""PLS regression, PLS canonical, CCA and PLS-SVD against scikit-learn: rotations, scores and predictions."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-7, "atol": 1e-9}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(17)
    x = g.standard_normal((30, 4))
    y = x @ g.standard_normal((4, 2)) + 0.3 * g.standard_normal((30, 2))
    pls = PLSRegression(n_components=2).fit(x, y)
    out.append(
        case("regression/rotations", "rotations", [x, y], np.asarray(pls.x_rotations_), **TOL, variant="regression")
    )
    out.append(case("regression/predict", "predict", [x, y], np.asarray(pls.predict(x)), **TOL, variant="regression"))
    canonical = PLSCanonical(n_components=2).fit(x, y)
    out.append(
        case("canonical/scores", "scores", [x, y], np.asarray(canonical.transform(x)), **TOL, variant="canonical")
    )
    c = CCA(n_components=2).fit(x, y)
    out.append(case("cca/scores", "scores", [x, y], np.asarray(c.transform(x)), **TOL, variant="cca"))
    s = PLSSVD(n_components=2).fit(x, y)
    out.append(case("svd/scores", "scores", [x, y], np.asarray(s.transform(x)), **TOL, variant="svd"))
    return out
