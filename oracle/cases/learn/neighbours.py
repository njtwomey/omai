"""Nearest-neighbour learners against sklearn.neighbors: the k-nearest classifier and regressor (uniform and distance
weights, several metrics), the radius classifier and regressor with an outlier label, and the nearest centroid
classifier with its priors, Manhattan medians and Tibshirani shrinkage."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestCentroid,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def blobs(g: np.random.Generator, n: int, d: int, k: int) -> tuple[FloatArray, FloatArray]:
    """Gaussian blobs with k classes, class means spread along the diagonal."""
    y = np.arange(n) % k
    x = g.normal(size=(n, d)) + 1.5 * y[:, None] * np.linspace(1, 0.4, d)[None, :]
    return x, y.astype(np.float64)


def cases() -> list[Case]:
    g = rng(2718)
    x, y = blobs(g, 60, 3, 3)
    xt = np.concatenate([g.normal(size=(12, 3)) + 1.5, x[:3] + 0.01])
    t = np.sin(x[:, 0]) + 0.5 * x[:, 1] + 0.1 * g.normal(size=60)
    out: list[Case] = []
    for metric, params in [
        ("euclidean", {}),
        ("manhattan", {}),
        ("chebyshev", {}),
        ("cosine", {}),
        ("minkowski", {"p": 3}),
    ]:
        for weights in ["uniform", "distance"]:
            clf = KNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric, algorithm="brute", **params).fit(
                x, y
            )
            out.append(
                case(
                    f"classifier/{metric}/{weights}",
                    "kClassifier",
                    [x, y, xt],
                    {"predict": clf.predict(xt), "proba": clf.predict_proba(xt)},
                    rtol=1e-9,
                    atol=1e-10,
                    metric=metric,
                    weights=weights,
                    k=5,
                    **params,
                )
            )
            reg = KNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric, algorithm="brute", **params).fit(
                x, t
            )
            out.append(
                case(
                    f"regressor/{metric}/{weights}",
                    "kRegressor",
                    [x, t, xt],
                    {"predict": reg.predict(xt)},
                    rtol=1e-9,
                    atol=1e-10,
                    metric=metric,
                    weights=weights,
                    k=4,
                    **params,
                )
            )
    for algorithm in ["kd_tree", "ball_tree"]:
        clf = KNeighborsClassifier(n_neighbors=7, algorithm=algorithm, leaf_size=5).fit(x, y)
        out.append(
            case(
                f"classifier/{algorithm}",
                "kClassifier",
                [x, y, xt],
                {"predict": clf.predict(xt), "proba": clf.predict_proba(xt)},
                rtol=1e-9,
                atol=1e-10,
                metric="euclidean",
                weights="uniform",
                k=7,
                algorithm=algorithm.replace("_", "-"),
                leafSize=5,
            )
        )
    far = np.vstack([xt, [[40.0, 40.0, 40.0]]])
    for weights in ["uniform", "distance"]:
        rc = RadiusNeighborsClassifier(radius=1.6, weights=weights, outlier_label=-1, algorithm="brute").fit(x, y)
        out.append(
            case(
                f"radius-classifier/{weights}",
                "radiusClassifier",
                [x, y, far],
                {"predict": rc.predict(far), "proba": rc.predict_proba(far)},
                rtol=1e-9,
                atol=1e-10,
                radius=1.6,
                weights=weights,
                outlier=-1,
            )
        )
        rr = RadiusNeighborsRegressor(radius=1.6, weights=weights, algorithm="brute").fit(x, t)
        out.append(
            case(
                f"radius-regressor/{weights}",
                "radiusRegressor",
                [x, t, xt],
                {"predict": rr.predict(xt)},
                rtol=1e-9,
                atol=1e-10,
                radius=1.6,
                weights=weights,
            )
        )
    rc = RadiusNeighborsClassifier(radius=1.6, outlier_label="most_frequent", algorithm="brute").fit(x, y)
    out.append(
        case(
            "radius-classifier/most-frequent",
            "radiusClassifier",
            [x, y, far],
            {"predict": rc.predict(far)},
            rtol=1e-9,
            atol=1e-10,
            radius=1.6,
            weights="uniform",
            outlier="most-frequent",
        )
    )
    for metric, shrink, priors in [
        ("euclidean", None, "uniform"),
        ("euclidean", None, "empirical"),
        ("euclidean", 0.4, "uniform"),
        ("euclidean", 1.0, "empirical"),
        ("manhattan", None, "uniform"),
        ("manhattan", 0.3, "uniform"),
    ]:
        xi, yi = blobs(g, 61, 4, 3)
        yi = np.where(np.arange(61) < 12, 2.0, yi)  # unbalanced classes
        nc = NearestCentroid(metric=metric, shrink_threshold=shrink, priors=priors).fit(xi, yi)
        q = np.concatenate([g.normal(size=(8, 4)) + 1.5, xi[:4]])
        output: dict[str, FloatArray] = {
            "centroids": nc.centroids_,
            "predict": nc.predict(q),
            "withinStd": nc.within_class_std_dev_,
        }
        if metric == "euclidean":
            output["decision"] = nc.decision_function(q)
            output["proba"] = nc.predict_proba(q)
        params: dict[str, object] = {"metric": metric, "priors": priors}
        if shrink is not None:
            params["shrink"] = shrink
        out.append(
            case(
                f"centroid/{metric}/{shrink}/{priors}", "centroid", [xi, yi, q], output, rtol=1e-9, atol=1e-10, **params
            )
        )
    return out
