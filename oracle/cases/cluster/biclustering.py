"""Spectral biclustering against scikit-learn: the normalisations (exact), the consensus score (exact), and the
partitions of co-clustering and biclustering on planted biclusters (the exact SVD, ten k-means starts), compared as
labels numbered by first appearance, so up to the permutation of the clusters (singular vectors have no sign)."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (  # pyright: ignore[reportPrivateUsage]
    _bistochastic_normalize,  # pyright: ignore[reportPrivateUsage]
    _log_normalize,  # pyright: ignore[reportPrivateUsage]
    _scale_normalize,  # pyright: ignore[reportPrivateUsage]
)
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def canonical(labels: object) -> NDArray[np.float64]:
    """Labels renumbered by first appearance."""
    seen: dict[int, int] = {}
    return np.array([seen.setdefault(int(v), len(seen)) for v in np.asarray(labels)], dtype=np.float64)


def indicators(n: int, k: int, g: np.random.Generator) -> NDArray[np.float64]:
    """k random 0/1 rows of length n, none empty."""
    out = (g.random((k, n)) < 0.4).astype(np.float64)
    for row in out:
        if not row.any():
            row[g.integers(0, n)] = 1.0
    return out


type Planted = tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]


def stable_seed(
    make: Callable[[int], Planted],
    fit: Callable[[NDArray[np.float64], int], SpectralCoclustering | SpectralBiclustering],
) -> int:
    """The first data seed for which scikit-learn recovers the planted biclusters whatever its random state."""
    for seed in range(100):
        data, rows, cols = make(seed)
        if all(consensus_score(fit(data, rs).biclusters_, (rows, cols)) > 0.999 for rs in range(4)):
            return seed
    raise RuntimeError("no stable seed")


def coclustering(data: NDArray[np.float64], k: int, random_state: int) -> SpectralCoclustering:
    return SpectralCoclustering(n_clusters=k, svd_method="arpack", random_state=random_state).fit(data)


def biclustering(data: NDArray[np.float64], shape: tuple[int, int], random_state: int) -> SpectralBiclustering:
    options: dict[str, Any] = {
        "n_clusters": shape,
        "method": "log",
        "svd_method": "arpack",
        "n_components": 6,
        "n_best": 3,
        "random_state": random_state,
    }
    return SpectralBiclustering(**options).fit(data)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(31)
    x = g.uniform(0.5, 5.0, (9, 6))
    signed = x - 2.0
    out.append(case("scale", "scale", [x], _scale_normalize(x)[0], rtol=1e-12, atol=1e-14))
    out.append(case("scale_shifted", "scale", [signed], _scale_normalize(signed)[0], rtol=1e-12, atol=1e-14))
    out.append(case("bistochastic", "bistochastic", [x], _bistochastic_normalize(x), rtol=1e-10, atol=1e-12))
    out.append(case("log", "log", [x], _log_normalize(x), rtol=1e-12, atol=1e-13))
    out.append(case("log_shifted", "log", [signed], _log_normalize(signed), rtol=1e-12, atol=1e-13))

    for na, nb in [(3, 3), (4, 2), (2, 5)]:
        a = (indicators(12, na, g), indicators(8, na, g))
        b = (indicators(12, nb, g), indicators(8, nb, g))
        out.append(
            case(
                f"consensus/{na}x{nb}",
                "consensus",
                [a[0], a[1], b[0], b[1]],
                np.float64(consensus_score(a, b)),
                rtol=1e-12,
                atol=1e-14,
            )
        )

    for k in [3, 4, 5]:
        seed = stable_seed(
            lambda s, k=k: make_biclusters((60, 40), k, noise=1.0, random_state=s),
            lambda data, rs, k=k: coclustering(data, k, rs),
        )
        data, _, _ = make_biclusters((60, 40), k, noise=1.0, random_state=seed)
        model = coclustering(data, k, 0)
        out.append(case(f"coclustering/{k}/rows", "coclusteringRows", [data], canonical(model.row_labels_), k=k))
        out.append(
            case(f"coclustering/{k}/columns", "coclusteringColumns", [data], canonical(model.column_labels_), k=k)
        )
        out.append(
            case(
                f"coclustering/{k}/consensus",
                "coclusteringConsensus",
                [data, model.rows_.astype(np.float64), model.columns_.astype(np.float64)],
                np.float64(1.0),
                rtol=1e-12,
                atol=1e-12,
                k=k,
            )
        )

    for shape in [(2, 3), (3, 3), (2, 2)]:
        seed = stable_seed(
            lambda s, sh=shape: make_checkerboard((60, 45), sh, noise=1.0, random_state=s),
            lambda data, rs, sh=shape: biclustering(data, sh, rs),
        )
        data, _, _ = make_checkerboard((60, 45), shape, noise=1.0, random_state=seed)
        model = biclustering(data, shape, 0)
        name = f"biclustering/{shape[0]}x{shape[1]}"
        options = {"kr": shape[0], "kc": shape[1]}
        out.append(case(f"{name}/rows", "biclusteringRows", [data], canonical(model.row_labels_), **options))
        out.append(case(f"{name}/columns", "biclusteringColumns", [data], canonical(model.column_labels_), **options))
    return out
