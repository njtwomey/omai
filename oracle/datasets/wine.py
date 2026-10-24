"""The UCI wine recognition data, from scikit-learn."""

from typing import cast

import sklearn
from sklearn.datasets import load_wine
from sklearn.utils import Bunch
from whitebox_oracle.vendor import render_classification

TARGET = "packages/engine/src/methods/datasets/_data/wine.data.ts"


def render() -> str:
    bunch = cast(Bunch, load_wine())
    return render_classification(
        "WINE",
        "wine",
        f"scikit-learn {sklearn.__version__} (load_wine)",
        bunch.data,
        bunch.target,
        list(bunch.feature_names),
        list(bunch.target_names),
    )
