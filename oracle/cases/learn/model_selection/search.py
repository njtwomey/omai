"""Hyperparameter searches against sklearn.model_selection: GridSearchCV over a pipeline (mean, deviation and rank of
the test scores, split scores, best candidate and its refitted predictions), a classifier grid with tied ranks, and
RandomizedSearchCV's evaluation of its sampled candidates."""

from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(4338)
    n = 40
    x = g.uniform(-2, 2, size=(n, 2))
    y = x[:, 0] ** 2 - x[:, 1] + 0.3 * g.normal(size=n)
    z = g.uniform(-2, 2, size=(5, 2))
    out: list[Case] = []
    pipe = Pipeline([("poly", PolynomialFeatures()), ("scale", StandardScaler()), ("ridge", Ridge())])
    grid = {"poly__degree": [1, 2, 3], "ridge__alpha": [0.1, 1.0, 10.0]}
    gs = GridSearchCV(pipe, grid, cv=KFold(4), scoring="neg_mean_squared_error", return_train_score=True).fit(x, y)
    res: dict[str, Any] = gs.cv_results_
    splits = np.array([res[f"split{i}_test_score"] for i in range(4)]).T
    out.append(
        case(
            "grid/pipeline",
            "gridPipeline",
            [x, y, z],
            {
                "mean": res["mean_test_score"],
                "std": res["std_test_score"],
                "rank": res["rank_test_score"].astype(np.int32),
                "splits": splits,
                "meanTrain": res["mean_train_score"],
                "bestIndex": np.array(gs.best_index_),
                "predict": gs.predict(z),
            },
            rtol=1e-9,
            atol=1e-12,
        )
    )
    xc = g.normal(size=(30, 2))
    yc = (xc[:, 0] * xc[:, 1] > 0).astype(np.float64)
    gs = GridSearchCV(
        KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9]}, cv=StratifiedKFold(3), scoring="accuracy"
    ).fit(xc, yc)
    res = gs.cv_results_
    out.append(
        case(
            "grid/knn",
            "gridKnn",
            [xc, yc],
            {
                "mean": res["mean_test_score"],
                "rank": res["rank_test_score"].astype(np.int32),
                "bestIndex": np.array(gs.best_index_),
            },
            rtol=1e-12,
            atol=1e-12,
        )
    )
    rs = RandomizedSearchCV(
        pipe,
        {"poly__degree": [1, 2, 3, 4], "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        n_iter=6,
        cv=KFold(3),
        scoring="r2",
        random_state=0,
    ).fit(x, y)
    res = rs.cv_results_
    degrees = [float(p["poly__degree"]) for p in res["params"]]
    alphas = [float(p["ridge__alpha"]) for p in res["params"]]
    out.append(
        case(
            "randomized/pipeline",
            "randomizedPipeline",
            [x, y, np.array(degrees), np.array(alphas)],
            {"mean": res["mean_test_score"], "rank": res["rank_test_score"].astype(np.int32)},
            rtol=1e-9,
            atol=1e-12,
        )
    )
    return out
