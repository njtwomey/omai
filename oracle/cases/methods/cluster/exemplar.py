"""Affinity propagation against scikit-learn: exemplars and labels on separated blobs."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.cluster import AffinityPropagation

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(41)
    x = np.concatenate([g.normal(c, 0.6, (12, 2)) for c in [(0, 0), (5, 1), (1, 6)]])
    out: list[Case] = []
    for pref in [None, -30.0]:
        ap = AffinityPropagation(preference=pref, random_state=0).fit(x)
        name = "median" if pref is None else "fixed"
        opts = {} if pref is None else {"preference": pref}
        out.append(
            case(f"{name}/centres", "centres", [x], np.asarray(ap.cluster_centers_indices_, dtype=np.float64), **opts)
        )
        out.append(case(f"{name}/labels", "labels", [x], np.asarray(ap.labels_, dtype=np.float64), **opts))
    return out
