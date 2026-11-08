"""Neighbour search and vector quantisation against sklearn.neighbors and sklearn.cluster.KMeans."""

from typing import Literal

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors, kneighbors_graph, radius_neighbors_graph

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}

TREE_METRICS: list[Literal["euclidean", "manhattan", "chebyshev"]] = ["euclidean", "manhattan", "chebyshev"]
OURS: dict[str, str] = {"euclidean": "euclidean", "manhattan": "cityblock", "chebyshev": "chebyshev"}


def cases() -> list[Case]:
    g = rng(170)
    out: list[Case] = []
    data = g.normal(size=(200, 3))
    queries = g.normal(size=(15, 3))
    nn = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(data)
    dist, idx = nn.kneighbors(queries)
    out.append(case("brute/indices", "bruteIndices", [data, queries], idx.astype(np.float64), rtol=0, k=5))
    out.append(case("brute/distances", "bruteDistances", [data, queries], dist, rtol=1e-12, k=5))
    small = data[:30]
    out.append(
        case(
            "kneighborsGraph/distance",
            "knnGraph",
            [small],
            kneighbors_graph(small, 3, mode="distance").toarray(),
            rtol=1e-12,
            k=3,
        )
    )
    out.append(
        case(
            "radiusGraph",
            "radiusGraph",
            [small],
            radius_neighbors_graph(small, 0.8, mode="distance").toarray(),
            rtol=1e-12,
            r=0.8,
        )
    )
    for metric in TREE_METRICS:
        kd = KDTree(data, leaf_size=8, metric=metric)
        dk, ik = kd.query(queries, k=4)
        out.append(
            case(
                f"kd/{metric}/indices", "kdIndices", [data, queries], ik.astype(np.float64), rtol=0, metric=OURS[metric]
            )
        )
        out.append(case(f"kd/{metric}/distances", "kdDistances", [data, queries], dk, rtol=1e-12, metric=OURS[metric]))
        counts = kd.query_radius(queries, r=1.0, count_only=True)
        out.append(
            case(
                f"kd/{metric}/counts",
                "kdCounts",
                [data, queries],
                np.asarray(counts, dtype=np.float64),
                rtol=0,
                metric=OURS[metric],
            )
        )
        bt = BallTree(data, leaf_size=8, metric=metric)
        db, ib = bt.query(queries, k=4)
        out.append(
            case(
                f"ball/{metric}/indices",
                "ballIndices",
                [data, queries],
                ib.astype(np.float64),
                rtol=0,
                metric=OURS[metric],
            )
        )
        out.append(
            case(f"ball/{metric}/distances", "ballDistances", [data, queries], db, rtol=1e-12, metric=OURS[metric])
        )
    geo = np.radians(g.uniform([-60, -150], [60, 150], size=(100, 2)))
    gq = np.radians(g.uniform([-60, -150], [60, 150], size=(8, 2)))
    hb = BallTree(geo, leaf_size=5, metric="haversine")
    dh, ih = hb.query(gq, k=3)
    out.append(
        case("ball/haversine/indices", "ballIndices", [geo, gq], ih.astype(np.float64), rtol=0, metric="haversine")
    )
    out.append(case("ball/haversine/distances", "ballDistances", [geo, gq], dh, rtol=1e-12, metric="haversine"))

    blobs = np.concatenate([g.normal(loc=c, scale=0.4, size=(40, 2)) for c in ([0, 0], [3, 3], [-3, 3])])
    init = blobs[[0, 40, 80]] + 0.5
    # The installed stubs type init and n_init as str; sklearn accepts an array and an int.
    km = KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(blobs)  # pyright: ignore[reportArgumentType]
    out.append(case("kmeans/centroids", "kmeansCentroids", [blobs, init], km.cluster_centers_, rtol=1e-10, atol=1e-12))
    out.append(case("kmeans/inertia", "kmeansInertia", [blobs, init], np.array(km.inertia_), rtol=1e-10))
    return out
