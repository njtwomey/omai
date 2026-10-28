"""Non-metric SMACOF from a fixed start against scikit-learn: the configuration and its raw stress, from dissimilarities
that are a monotone transform of Euclidean distances (so isotonic regression has ties and violations to resolve)."""

from importlib.metadata import version
from typing import Any, cast

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import smacof

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(15)
    x = g.standard_normal((16, 3))
    d = squareform(np.round(pdist(x) ** 1.5 + 0.1 * g.standard_normal(120), 1).clip(0.1))
    init = g.uniform(0, 1, (16, 2))
    options: dict[str, Any] = {"init": init, "n_init": 1, "metric": False, "max_iter": 200, "eps": 1e-6}
    result = smacof(d, normalized_stress=cast(Any, False), **options)
    y, stress = result[0], result[1]
    return [
        case("nonmetric/embedding", "embedding", [d, init], np.asarray(y), rtol=1e-7, atol=1e-8),
        case("nonmetric/stress", "stress", [d, init], np.float64(stress), rtol=1e-7, atol=1e-8),
    ]
