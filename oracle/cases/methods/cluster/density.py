"""Density clustering against scikit-learn: the OPTICS ordering (reachability, core distances, order) and labels."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from numpy.typing import ArrayLike
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def finite(v: ArrayLike) -> FloatArray:
    a = np.asarray(v, dtype=np.float64)
    return np.where(np.isfinite(a), a, -1.0)


def canonical(labels: ArrayLike) -> FloatArray:
    """Labels renumbered by first appearance (noise stays -1), so equal partitions compare equal."""
    seen: dict[int, int] = {}
    out: list[float] = []
    for v in np.asarray(labels).tolist():
        if v < 0:
            out.append(-1.0)
            continue
        seen.setdefault(int(v), len(seen))
        out.append(float(seen[int(v)]))
    return np.asarray(out, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(11)
    x = np.concatenate([g.normal(0, 0.4, (20, 2)), g.normal(3, 0.6, (25, 2)), g.uniform(-2, 5, (5, 2))])
    for min_samples in [3, 5]:
        o = OPTICS(min_samples=min_samples, max_eps=np.inf, cluster_method="dbscan", eps=0.5).fit(x)
        opts = {"minSamples": min_samples}
        out.append(
            case(
                f"optics/{min_samples}/reachability",
                "reachability",
                [x],
                finite(o.reachability_),
                rtol=1e-10,
                atol=1e-10,
                **opts,
            )
        )
        out.append(
            case(f"optics/{min_samples}/core", "core", [x], finite(o.core_distances_), rtol=1e-10, atol=1e-10, **opts)
        )
        out.append(
            case(f"optics/{min_samples}/ordering", "ordering", [x], np.asarray(o.ordering_, dtype=np.float64), **opts)
        )
        out.append(
            case(f"optics/{min_samples}/labels", "opticsLabels", [x], np.asarray(o.labels_, dtype=np.float64), **opts)
        )
    d = DBSCAN(eps=0.5, min_samples=4).fit(x)
    out.append(case("dbscan/labels", "dbscan", [x], np.asarray(d.labels_, dtype=np.float64), eps=0.5, minSamples=4))
    h = HDBSCAN(min_cluster_size=5).fit(x)
    out.append(case("hdbscan/labels", "hdbscan", [x], canonical(h.labels_), minClusterSize=5))
    return out
