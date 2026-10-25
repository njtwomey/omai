"""Local outlier factors against scikit-learn: training LOFs and novelty scores of new points."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.neighbors import LocalOutlierFactor

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-10}


def cases() -> list[Case]:
    g = rng(29)
    x = np.concatenate([g.normal(0, 1, (40, 2)), g.uniform(-6, 6, (5, 2))])
    q = g.uniform(-4, 4, (6, 2))
    out: list[Case] = []
    for k in [5, 20]:
        lof = LocalOutlierFactor(n_neighbors=k).fit(x)
        out.append(case(f"lof{k}/train", "train", [x], -np.asarray(lof.negative_outlier_factor_), **TOL, k=k))
        nov = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(x)
        out.append(case(f"lof{k}/novel", "novel", [x, q], -np.asarray(nov.score_samples(q)), **TOL, k=k))
    return out
