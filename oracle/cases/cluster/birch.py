"""BIRCH against scikit-learn: the leaf subcluster centres in chain order (so the splits of the CF tree are exercised
by a small branching factor) and the labels after the global Ward step."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from numpy import typing as npt
from sklearn.cluster import Birch

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def canonical(labels: npt.ArrayLike) -> FloatArray:
    """Labels renumbered by first appearance, so equal partitions compare equal."""
    seen: dict[int, int] = {}
    out: list[float] = []
    for v in np.asarray(labels).tolist():
        seen.setdefault(int(v), len(seen))
        out.append(float(seen[int(v)]))
    return np.asarray(out, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(5)
    x = np.concatenate([g.normal(c, 0.7, (25, 2)) for c in [(0, 0), (6, 1), (2, 7)]])
    g.shuffle(x)
    out: list[Case] = []
    for threshold, branching in [(0.5, 3), (0.8, 4), (1.5, 50)]:
        b = Birch(threshold=threshold, branching_factor=branching, n_clusters=3).fit(x)
        opts = {"threshold": threshold, "branchingFactor": branching}
        name = f"{threshold}/{branching}"
        out.append(
            case(
                f"{name}/subclusters",
                "subclusters",
                [x],
                np.asarray(b.subcluster_centers_, dtype=np.float64),
                rtol=1e-10,
                atol=1e-10,
                **opts,
            )
        )
        out.append(case(f"{name}/labels", "labels", [x], canonical(b.labels_), **opts))
    return out
