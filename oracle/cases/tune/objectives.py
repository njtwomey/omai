"""The default cross-validation folds of the estimator objectives against scikit-learn: `KFold(k)` for regression and
`StratifiedKFold(k)` for classification, both unshuffled, as `cv=<int>` chooses them."""

import numpy as np
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from whitebox_oracle.fixture import Case, Json, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def kfold_case(name: str, n: int, k: int) -> Case:
    tests: list[Json] = [[int(i) for i in test] for _, test in KFold(k).split(np.zeros((n, 1)))]
    return case(name, "kFold", [], tests, n=n, k=k)


def stratified_case(name: str, labels: list[int], k: int) -> Case:
    y = np.asarray(labels)
    tests: list[Json] = [[int(i) for i in test] for _, test in StratifiedKFold(k).split(np.zeros((len(y), 1)), y)]
    return case(name, "stratifiedKFold", [labels], tests, k=k)


def cases() -> list[Case]:
    g = rng(0)
    return [
        kfold_case("kFold/10/3", 10, 3),
        kfold_case("kFold/23/5", 23, 5),
        kfold_case("kFold/5/5", 5, 5),
        stratified_case("stratified/balanced", [0, 1] * 10, 5),
        stratified_case("stratified/unbalanced", [int(v) for v in g.choice(3, size=37, p=[0.6, 0.3, 0.1])], 4),
        stratified_case("stratified/first-seen", [2, 2, 0, 1, 1, 0, 2, 0, 1, 2, 1, 1, 0, 2, 2], 3),
    ]
