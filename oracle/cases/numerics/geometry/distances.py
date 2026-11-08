"""Distances against scipy.spatial.distance and sklearn.metrics.pairwise.haversine_distances."""

from typing import Literal

import numpy as np
import scipy
import scipy.spatial.distance as sd
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.metrics.pairwise import haversine_distances

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sklearn": sklearn.__version__}

type Plain = Literal[
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "seuclidean",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "mahalanobis",
]
METRICS: list[Plain] = [
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "seuclidean",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "mahalanobis",
]


def cases() -> list[Case]:
    g = rng(150)
    out: list[Case] = []
    xa = g.normal(size=(6, 4))
    xb = g.normal(size=(5, 4))
    for metric in METRICS:
        out.append(
            case(f"cdist/{metric}", "cdist", [xa, xb], sd.cdist(xa, xb, metric), rtol=1e-11, atol=1e-12, metric=metric)
        )
        out.append(case(f"pdist/{metric}", "pdist", [xa], sd.pdist(xa, metric), rtol=1e-11, atol=1e-12, metric=metric))
    for p in (1.0, 3.0, 0.5):
        out.append(
            case(
                f"cdist/minkowski/{p}",
                "cdist",
                [xa, xb],
                sd.cdist(xa, xb, "minkowski", p=p),
                rtol=1e-11,
                metric="minkowski",
                p=p,
            )
        )
    w = g.uniform(0.5, 2, size=4)
    out.append(
        case("cdist/weighted", "cdistWeighted", [xa, xb, w], sd.cdist(xa, xb, "minkowski", p=3.0, w=w), rtol=1e-11)
    )
    ba = (g.uniform(size=(6, 8)) > 0.5).astype(np.float64)
    bb = (g.uniform(size=(5, 8)) > 0.5).astype(np.float64)
    for metric in ("hamming", "jaccard"):
        out.append(
            case(
                f"cdist/{metric}",
                "cdist",
                [ba, bb],
                sd.cdist(ba.astype(bool), bb.astype(bool), metric),
                rtol=1e-13,
                metric=metric,
            )
        )
    out.append(case("squareform", "squareform", [sd.pdist(xa)], sd.squareform(sd.pdist(xa)), rtol=1e-14))
    geo = np.radians(g.uniform([-80, -170], [80, 170], size=(5, 2)))
    out.append(
        case(
            "haversine",
            "cdist",
            [geo, geo[:3]],
            haversine_distances(geo, geo[:3]),
            rtol=1e-12,
            atol=1e-14,
            metric="haversine",
        )
    )
    return out
