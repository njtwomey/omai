"""Nearest centroids against scikit-learn, plain and shrunken."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.neighbors import NearestCentroid

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(62)
    y = np.arange(45) % 3
    x = g.normal(size=(45, 4)) + np.array([[0, 0, 0, 0], [1, 0.2, 0, 0], [0, 1, 0.1, 0]])[y]
    out: list[Case] = []
    for shrink in [0.0, 0.5]:
        nc = NearestCentroid(shrink_threshold=shrink if shrink > 0 else None).fit(x, y)
        out.append(
            case(
                f"centroid/{shrink}",
                "centroid",
                [x, y.astype(np.float64)],
                {"centroids": nc.centroids_, "predict": nc.predict(x[:10]).astype(np.float64)},
                rtol=1e-10,
                atol=1e-12,
                shrink=shrink,
            )
        )
    return out
