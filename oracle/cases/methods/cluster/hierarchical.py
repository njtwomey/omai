"""Agglomerative clustering against scipy: linkage matrices for every linkage, cophenetic distances and flat cuts."""

from importlib.metadata import version
from typing import Literal

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.cluster.hierarchy import cophenet, linkage

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "scikit-learn": version("scikit-learn"),
}

METHODS: list[Literal["single", "complete", "average", "weighted", "ward", "centroid", "median"]] = [
    "single",
    "complete",
    "average",
    "weighted",
    "ward",
    "centroid",
    "median",
]


def cases() -> list[Case]:
    out: list[Case] = []
    x = rng(7).standard_normal((24, 3))
    for method in METHODS:
        z = linkage(x, method=method)
        out.append(case(f"linkage/{method}", "linkage", [x], z, rtol=1e-10, atol=1e-10, method=method))
    z = linkage(x, method="average")
    out.append(case("cophenet/average", "cophenet", [x], np.asarray(cophenet(z)), rtol=1e-10, atol=1e-10))
    return out
