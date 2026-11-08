"""Isomap against scikit-learn, compared through the pairwise distances of the embedding (sign-free)."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.spatial.distance import pdist
from sklearn.manifold import Isomap

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def roll(n: int) -> FloatArray:
    g = rng(12)
    t = 1.5 * np.pi * (1 + 2 * g.uniform(0, 1, n))
    return np.c_[t * np.cos(t), 10 * g.uniform(0, 1, n), t * np.sin(t)]


def cases() -> list[Case]:
    x = roll(50)
    e = Isomap(n_neighbors=8, n_components=2).fit_transform(x)
    return [case("isomap/pdist", "isomap", [x], pdist(e), rtol=1e-8, atol=1e-8, k=8)]
