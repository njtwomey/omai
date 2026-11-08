"""Spectral embedding against scikit-learn, compared through the pairwise distances of the embedding (sign-free)."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.spatial.distance import pdist
from sklearn.manifold import SpectralEmbedding

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(13)
    x = np.concatenate([g.normal(0, 1, (25, 3)), g.normal(4, 1, (25, 3))])
    out: list[Case] = []
    e = SpectralEmbedding(
        n_components=2, affinity="rbf", gamma=0.2, eigen_solver="lobpcg", random_state=0
    ).fit_transform(x)
    out.append(case("rbf/pdist", "rbf", [x], pdist(e), rtol=1e-5, atol=1e-7, gamma=0.2))
    return out
