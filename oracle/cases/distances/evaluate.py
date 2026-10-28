"""Named distances and similarities against scipy.spatial.distance (cdist for every registered distance with its
parameters, and scipy's scalar distance functions row by row for the elementwise form), scikit-learn's pairwise
kernels (cosine_similarity, linear_kernel, rbf_kernel, laplacian_kernel) and haversine_distances on degrees, and
numpy's corrcoef for the Pearson similarity. The registered name travels in the `metric` parameter (`name` is the
case's own name) and the registry's options beside it, tensors included."""

from collections.abc import Callable
from typing import Literal, TypedDict

import numpy as np
import scipy
import scipy.spatial.distance as sd
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng, spd
from sklearn.metrics.pairwise import (
    cosine_similarity,
    haversine_distances,
    laplacian_kernel,
    linear_kernel,
    rbf_kernel,
)

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__}

type MetricName = Literal[
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "minkowski",
    "seuclidean",
    "cosine",
    "correlation",
    "mahalanobis",
    "canberra",
    "braycurtis",
    "hamming",
    "jaccard",
]


class Options(TypedDict, total=False):
    """The registry's options of a case: scalars or tensors, in the registry's names."""

    p: float
    weights: FloatArray
    variance: FloatArray
    inverseCovariance: FloatArray


class ScipyKwargs(TypedDict, total=False):
    """scipy's keyword arguments for the same parameters."""

    p: float
    w: FloatArray
    V: FloatArray
    VI: FloatArray


# Registered name, scipy's metric name, and the options each case passes.
NAMED: list[tuple[str, MetricName, Options]] = [
    ("euclidean", "euclidean", {}),
    ("squaredEuclidean", "sqeuclidean", {}),
    ("manhattan", "cityblock", {}),
    ("chebyshev", "chebyshev", {}),
    ("minkowski", "minkowski", {"p": 3.0}),
    ("minkowski", "minkowski", {"p": 1.5}),
    ("seuclidean", "seuclidean", {}),
    ("cosine", "cosine", {}),
    ("correlation", "correlation", {}),
    ("mahalanobis", "mahalanobis", {}),
    ("canberra", "canberra", {}),
    ("braycurtis", "braycurtis", {}),
]


def scipy_kwargs(options: Options) -> ScipyKwargs:
    out: ScipyKwargs = {}
    if "p" in options:
        out["p"] = float(options["p"])
    if "weights" in options:
        out["w"] = np.asarray(options["weights"], dtype=np.float64)
    if "variance" in options:
        out["V"] = np.asarray(options["variance"], dtype=np.float64)
    if "inverseCovariance" in options:
        out["VI"] = np.asarray(options["inverseCovariance"], dtype=np.float64)
    return out


def pairwise(xa: FloatArray, xb: FloatArray, metric: MetricName, options: Options) -> FloatArray:
    return np.asarray(sd.cdist(xa, xb, metric, **scipy_kwargs(options)), dtype=np.float64)


def rowwise(metric: MetricName, xa: FloatArray, xb: FloatArray, options: Options) -> FloatArray:
    """scipy's scalar distance between corresponding rows, with cdist's defaults for the statistics."""
    f: Callable[..., float] = getattr(sd, metric)
    kwargs: dict[str, object] = dict(scipy_kwargs(options))
    stacked = np.vstack([xa, xb])
    if metric == "seuclidean" and "V" not in kwargs:
        kwargs["V"] = np.var(stacked, axis=0, ddof=1)
    if metric == "mahalanobis" and "VI" not in kwargs:
        kwargs["VI"] = np.linalg.inv(np.cov(stacked.T))
    return np.asarray([f(u, v, **kwargs) for u, v in zip(xa, xb, strict=True)], dtype=np.float64)


def label(name: str, options: Options) -> str:
    return "/".join([name, *[f"{k}{v}" for k, v in options.items() if isinstance(v, float)]])


def cases() -> list[Case]:
    g = rng(151)
    out: list[Case] = []
    xa = g.normal(size=(6, 4))
    xb = g.normal(size=(5, 4))
    rows_b = g.normal(size=(6, 4))
    for name, metric, options in NAMED:
        tol = 1e-10 if metric == "mahalanobis" else 1e-11
        out.append(
            case(
                f"pairwise/{label(name, options)}",
                "pairwise",
                [xa, xb],
                pairwise(xa, xb, metric, options),
                rtol=tol,
                atol=1e-12,
                metric=name,
                **options,
            )
        )
        out.append(
            case(
                f"elementwise/{label(name, options)}",
                "elementwise",
                [xa, rows_b],
                rowwise(metric, xa, rows_b, options),
                rtol=tol,
                atol=1e-12,
                metric=name,
                **options,
            )
        )
    # Explicit tensor parameters: weights, variances, a covariance and its inverse.
    w = g.uniform(0.5, 2, size=4)
    weighted: list[tuple[str, MetricName, Options]] = [
        ("euclidean", "euclidean", {"weights": w}),
        ("manhattan", "cityblock", {"weights": w}),
        ("minkowski", "minkowski", {"weights": w, "p": 3.0}),
    ]
    for name, metric, options in weighted:
        out.append(
            case(
                f"pairwise/{name}/weighted",
                "pairwise",
                [xa, xb],
                pairwise(xa, xb, metric, options),
                rtol=1e-11,
                metric=name,
                **options,
            )
        )
    v = g.uniform(0.5, 2, size=4)
    out.append(
        case(
            "pairwise/seuclidean/variance",
            "pairwise",
            [xa, xb],
            pairwise(xa, xb, "seuclidean", {"variance": v}),
            rtol=1e-11,
            metric="seuclidean",
            variance=v,
        )
    )
    cov = spd(g, 4)
    vi = np.linalg.inv(cov)
    out.append(
        case(
            "pairwise/mahalanobis/covariance",
            "pairwise",
            [xa, xb],
            pairwise(xa, xb, "mahalanobis", {"inverseCovariance": vi}),
            rtol=1e-10,
            metric="mahalanobis",
            covariance=cov,
        )
    )
    out.append(
        case(
            "elementwise/mahalanobis/inverseCovariance",
            "elementwise",
            [xa, rows_b],
            rowwise("mahalanobis", xa, rows_b, {"inverseCovariance": vi}),
            rtol=1e-10,
            metric="mahalanobis",
            inverseCovariance=vi,
        )
    )
    # Binary data for hamming and jaccard (scipy's boolean forms).
    ba = (g.uniform(size=(6, 8)) > 0.5).astype(np.float64)
    bb = (g.uniform(size=(5, 8)) > 0.5).astype(np.float64)
    bc = (g.uniform(size=(6, 8)) > 0.5).astype(np.float64)
    binary: list[MetricName] = ["hamming", "jaccard"]
    for metric in binary:
        out.append(
            case(
                f"pairwise/{metric}",
                "pairwise",
                [ba, bb],
                np.asarray(sd.cdist(ba.astype(bool), bb.astype(bool), metric), dtype=np.float64),
                rtol=1e-13,
                metric=metric,
            )
        )
        out.append(
            case(
                f"elementwise/{metric}",
                "elementwise",
                [ba, bc],
                rowwise(metric, ba.astype(bool), bc.astype(bool), {}),
                rtol=1e-13,
                metric=metric,
            )
        )
    # Haversine on degrees (sklearn takes radians), the central angle and scaled to kilometres on Earth.
    geo_a = g.uniform([-80, -170], [80, 170], size=(5, 2))
    geo_b = g.uniform([-80, -170], [80, 170], size=(3, 2))
    angle = haversine_distances(np.radians(geo_a), np.radians(geo_b))
    out.append(
        case("pairwise/haversine", "pairwise", [geo_a, geo_b], angle, rtol=1e-12, atol=1e-14, metric="haversine")
    )
    out.append(
        case(
            "pairwise/haversine/km",
            "pairwise",
            [geo_a, geo_b],
            6371.0 * angle,
            rtol=1e-12,
            atol=1e-11,
            metric="haversine",
            radius=6371.0,
        )
    )
    out.append(
        case(
            "pairwise/haversine/radians",
            "pairwise",
            [np.radians(geo_a), np.radians(geo_b)],
            angle,
            rtol=1e-12,
            atol=1e-14,
            metric="haversine",
            degrees=False,
        )
    )
    out.append(
        case(
            "elementwise/haversine",
            "elementwise",
            [geo_a, geo_a[::-1]],
            np.diag(haversine_distances(np.radians(geo_a), np.radians(geo_a[::-1]))),
            rtol=1e-12,
            atol=1e-14,
            metric="haversine",
        )
    )
    # Similarities against scikit-learn and numpy.
    m = xa.shape[0]
    out.append(
        case("pairwise/cosine", "pairwiseSimilarity", [xa, xb], cosine_similarity(xa, xb), rtol=1e-11, metric="cosine")
    )
    out.append(case("pairwise/dot", "pairwiseSimilarity", [xa, xb], linear_kernel(xa, xb), rtol=1e-12, metric="dot"))
    out.append(
        case(
            "pairwise/pearson",
            "pairwiseSimilarity",
            [xa, xb],
            np.corrcoef(xa, xb)[:m, m:],
            rtol=1e-11,
            metric="pearson",
        )
    )
    out.append(
        case(
            "pairwise/gaussian",
            "pairwiseSimilarity",
            [xa, xb],
            rbf_kernel(xa, xb, gamma=0.7),
            rtol=1e-11,
            metric="gaussian",
            gamma=0.7,
        )
    )
    out.append(
        case(
            "pairwise/gaussian/bandwidth",
            "pairwiseSimilarity",
            [xa, xb],
            rbf_kernel(xa, xb, gamma=1 / (2 * 1.3**2)),
            rtol=1e-11,
            metric="gaussian",
            bandwidth=1.3,
        )
    )
    out.append(
        case(
            "pairwise/laplacian",
            "pairwiseSimilarity",
            [xa, xb],
            laplacian_kernel(xa, xb, gamma=0.3),
            rtol=1e-11,
            metric="laplacian",
            gamma=0.3,
        )
    )
    out.append(
        case(
            "pairwise/inverseDistance/manhattan",
            "pairwiseSimilarity",
            [xa, xb],
            1 / (1 + pairwise(xa, xb, "cityblock", {}) / 2.5),
            rtol=1e-11,
            metric="inverseDistance",
            distance="manhattan",
            scale=2.5,
        )
    )
    out.append(
        case(
            "elementwise/cosine",
            "elementwiseSimilarity",
            [xa, rows_b],
            1 - rowwise("cosine", xa, rows_b, {}),
            rtol=1e-11,
            metric="cosine",
        )
    )
    out.append(
        case(
            "elementwise/pearson",
            "elementwiseSimilarity",
            [xa, rows_b],
            np.asarray([np.corrcoef(u, v)[0, 1] for u, v in zip(xa, rows_b, strict=True)]),
            rtol=1e-11,
            metric="pearson",
        )
    )
    out.append(
        case(
            "elementwise/gaussian",
            "elementwiseSimilarity",
            [xa, rows_b],
            np.exp(-0.7 * rowwise("sqeuclidean", xa, rows_b, {})),
            rtol=1e-11,
            metric="gaussian",
            gamma=0.7,
        )
    )
    out.append(
        case(
            "elementwise/inverseDistance/minkowski",
            "elementwiseSimilarity",
            [xa, rows_b],
            1 / (1 + rowwise("minkowski", xa, rows_b, {"p": 3.0})),
            rtol=1e-11,
            metric="inverseDistance",
            distance="minkowski",
            p=3.0,
        )
    )
    return out
