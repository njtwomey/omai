"""Metric SMACOF from a fixed start against scikit-learn: the configuration and its raw stress."""

from importlib.metadata import version
from typing import Any, cast

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import smacof

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(14)
    x = g.standard_normal((20, 4))
    d = squareform(pdist(x))
    init = g.uniform(0, 1, (20, 2))
    options: dict[str, Any] = {"init": init, "n_init": 1, "metric": True, "max_iter": 300, "eps": 1e-6}
    result = smacof(d, normalized_stress=cast(Any, False), **options)
    y, stress = result[0], result[1]
    return [
        case("smacof/embedding", "embedding", [x, init], np.asarray(y), rtol=1e-8, atol=1e-9),
        case("smacof/stress", "stress", [x, init], np.float64(stress), rtol=1e-8, atol=1e-9),
    ]
