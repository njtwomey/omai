"""The search wrappers against scikit-learn: `GridSearchCV` of a ridge regression over a grid (and over a list of
grids) with unshuffled k-fold, compared by the order of candidates, the mean and standard deviation of the fold
scores (mean squared error, which scikit-learn reports negated), their ranks and the best candidate."""

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from whitebox_oracle.fixture import Case, Json, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def search_case(name: str, grids: list[dict[str, list[float | bool]]], seed: int, cv: int) -> Case:
    g = rng(seed)
    x = g.normal(size=(30, 3))
    y = x @ np.asarray([1.0, -2.0, 0.5]) + 0.3 * g.normal(size=30) + 2.0
    search = GridSearchCV(Ridge(), grids, cv=cv, scoring="neg_mean_squared_error").fit(x, y)
    res = search.cv_results_
    out: dict[str, Json] = {
        "params": [{k: v for k, v in p.items()} for p in res["params"]],
        "mean": [-float(v) for v in res["mean_test_score"]],
        "std": [float(v) for v in res["std_test_score"]],
        "rank": [int(v) for v in res["rank_test_score"]],
        "bestIndex": int(search.best_index_),
    }
    inputs: list[Json] = [{k: list(v) for k, v in grid.items()} for grid in grids]
    return case(name, "gridSearchCv", [x, y, inputs], out, rtol=1e-9, atol=1e-12, cv=cv)


def cases() -> list[Case]:
    return [
        search_case("gridSearchCv/alpha", [{"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}], 0, 5),
        search_case(
            "gridSearchCv/list",
            [{"alpha": [0.1, 10.0], "fit_intercept": [True]}, {"alpha": [1.0], "fit_intercept": [False, True]}],
            1,
            3,
        ),
    ]
