"""Internal cluster-validity indices against sklearn.metrics; the Dunn index from its definition."""

from typing import Literal

import numpy as np
import scipy.spatial.distance as ssd
import sklearn
import sklearn.metrics as M
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def dunn(x: FloatArray, labels: FloatArray, metric: Literal["euclidean", "cityblock"]) -> float:
    d = ssd.squareform(ssd.pdist(x, metric))
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(x), dtype=bool)
    return float(d[~same].min() / d[same & off].max())


def cases() -> list[Case]:
    g = rng(62)
    centres = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 5.0]])
    labels = g.integers(0, 3, size=45).astype(np.float64)
    x = centres[labels.astype(int)] + g.normal(size=(45, 2))
    d = ssd.squareform(ssd.pdist(x, "cityblock"))
    return [
        case("silhouetteScore", "silhouetteScore", [x, labels], np.asarray(M.silhouette_score(x, labels))),
        case(
            "silhouetteScore/cityblock",
            "silhouetteScore",
            [x, labels],
            np.asarray(M.silhouette_score(x, labels, metric="cityblock")),
            metric="cityblock",
        ),
        case(
            "silhouetteScore/precomputed",
            "silhouetteScore",
            [d, labels],
            np.asarray(M.silhouette_score(d, labels, metric="precomputed")),
            metric="precomputed",
        ),
        case("silhouetteSamples", "silhouetteSamples", [x, labels], M.silhouette_samples(x, labels)),
        case(
            "calinskiHarabaszScore",
            "calinskiHarabaszScore",
            [x, labels],
            np.asarray(M.calinski_harabasz_score(x, labels)),
        ),
        case("daviesBouldinScore", "daviesBouldinScore", [x, labels], np.asarray(M.davies_bouldin_score(x, labels))),
        case("dunnIndex", "dunnIndex", [x, labels], np.asarray(dunn(x, labels, "euclidean"))),
        case(
            "dunnIndex/cityblock",
            "dunnIndex",
            [x, labels],
            np.asarray(dunn(x, labels, "cityblock")),
            metric="cityblock",
        ),
    ]
