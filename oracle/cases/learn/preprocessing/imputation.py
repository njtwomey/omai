"""Imputers against sklearn.impute: simple (every strategy, empty features, indicators and the inverse), k-NN
(uniform and distance weights) and iterative imputation with a ridge regressor, plus the missing indicator."""

from typing import Any

import numpy as np
import sklearn
import sklearn.experimental.enable_iterative_imputer  # noqa: F401  # pyright: ignore[reportUnusedImport]  (registers IterativeImputer)
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(4334)
    n, d = 30, 4
    latent = g.normal(size=(n, 2))
    x = latent @ g.normal(size=(2, d)) + 0.3 * g.normal(size=(n, d))
    x[:, 3] = np.round(x[:, 3])
    mask = g.uniform(size=(n, d)) < 0.2
    mask[:, 0] = False
    mask[0] = [False, True, True, True]
    xm = np.where(mask, np.nan, x)
    z = np.round(x[:6] + 0.1 * g.normal(size=(6, d)), 3)
    z[[0, 2, 4], [1, 2, 3]] = np.nan
    z[5, 1:] = np.nan
    out: list[Case] = []
    for strategy in ["mean", "median", "most_frequent", "constant"]:
        est: Any = SimpleImputer(strategy=strategy, fill_value=-7.0, add_indicator=True).fit(xm)
        t = est.transform(z)
        out.append(
            case(
                f"simple/{strategy}",
                "simple",
                [xm, z],
                {"transform": t, "inverse": est.inverse_transform(t)},
                strategy=strategy,
                fillValue=-7.0,
            )
        )
    xe = np.column_stack([xm[:, :2], np.full(n, np.nan)])
    for keep in [False, True]:
        est = SimpleImputer(keep_empty_features=keep).fit(xe)
        out.append(
            case(
                f"simple/empty-{keep}",
                "simple",
                [xe, z[:, :3]],
                {"transform": est.transform(z[:, :3])},
                strategy="mean",
                keepEmptyFeatures=keep,
                addIndicator=False,
            )
        )
    for weights in ["uniform", "distance"]:
        for k in [1, 3, 5]:
            est = KNNImputer(n_neighbors=k, weights=weights).fit(xm)
            out.append(
                case(
                    f"knn/{weights}/{k}",
                    "knn",
                    [xm, z],
                    {"transform": est.transform(z), "train": est.transform(xm)},
                    nNeighbors=k,
                    weights=weights,
                    rtol=1e-10,
                    atol=1e-10,
                )
            )
    est = KNNImputer(n_neighbors=2, add_indicator=True).fit(xm)
    out.append(
        case(
            "knn/indicator",
            "knn",
            [xm, z],
            {"transform": est.transform(z)},
            nNeighbors=2,
            weights="uniform",
            addIndicator=True,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    for order in ["ascending", "descending", "roman", "arabic"]:
        for max_iter in [1, 10]:
            est = IterativeImputer(
                estimator=Ridge(alpha=1.0), max_iter=max_iter, tol=1e-3, imputation_order=order, random_state=0
            ).fit(xm)
            out.append(
                case(
                    f"iterative/{order}/{max_iter}",
                    "iterative",
                    [xm, z],
                    {"transform": est.transform(z), "nIter": np.array(est.n_iter_)},
                    order=order,
                    maxIter=max_iter,
                    rtol=1e-9,
                    atol=1e-10,
                )
            )
    est = IterativeImputer(
        estimator=Ridge(alpha=1.0),
        initial_strategy="median",
        min_value=-1.0,
        max_value=1.5,
        skip_complete=True,
        random_state=0,
    ).fit(xm)
    out.append(
        case(
            "iterative/clipped",
            "iterative",
            [xm, z],
            {"transform": est.transform(z)},
            order="ascending",
            maxIter=10,
            initialStrategy="median",
            minValue=-1.0,
            maxValue=1.5,
            skipComplete=True,
            rtol=1e-9,
            atol=1e-10,
        )
    )
    mi = MissingIndicator(features="all").fit(xm)
    out.append(
        case("indicator/all", "indicator", [xm, z], {"transform": mi.transform(z).astype(np.float64)}, features="all")
    )
    zm = np.where(np.isnan(z[:, :1]), 0.0, z[:, :1])
    xs = np.column_stack([zm, z[:, 1:]])
    mi = MissingIndicator().fit(xm)
    out.append(
        case(
            "indicator/missing-only",
            "indicator",
            [xm, xs],
            {"transform": mi.transform(xs).astype(np.float64)},
            features="missing-only",
        )
    )
    return out
