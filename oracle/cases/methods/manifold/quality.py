"""Trustworthiness against scikit-learn."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.manifold import trustworthiness

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(3)
    x = g.standard_normal((30, 4))
    e = x[:, :2] + 0.5 * g.standard_normal((30, 2))
    return [
        case(f"trustworthiness/{k}", "trustworthiness", [x, e], np.float64(trustworthiness(x, e, n_neighbors=k)), k=k)
        for k in [1, 3, 5, 10]
    ]
