"""k-means with fixed initial centres against scikit-learn's Lloyd iterations: final centres, labels and inertia."""

from importlib.metadata import version
from typing import Any

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.cluster import KMeans

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def blobs(seed: int, n: int, k: int, d: int) -> FloatArray:
    g = rng(seed)
    centres = g.uniform(-6, 6, size=(k, d))
    return np.concatenate([c + 0.8 * g.standard_normal((n // k, d)) for c in centres])


def cases() -> list[Case]:
    out: list[Case] = []
    for name, seed, k, d in [("blobs3", 1, 3, 2), ("blobs4", 2, 4, 3)]:
        x = blobs(seed, 60, k, d)
        init = x[rng(seed + 10).choice(len(x), size=k, replace=False)]
        options: dict[str, Any] = {"n_clusters": k, "init": init, "n_init": 1, "algorithm": "lloyd", "tol": 1e-10}
        km = KMeans(**options).fit(x)
        out.append(case(f"{name}/centres", "centres", [x, init], km.cluster_centers_, rtol=1e-10, atol=1e-10))
        out.append(case(f"{name}/labels", "labels", [x, init], np.asarray(km.labels_, dtype=np.float64)))
        out.append(case(f"{name}/inertia", "inertia", [x, init], np.float64(km.inertia_), rtol=1e-10, atol=1e-10))
    return out
