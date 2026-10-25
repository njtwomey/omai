"""The UCI wine recognition data, from scikit-learn."""

from typing import cast

import sklearn
from mdml_oracle.vendor import render_classification
from sklearn.datasets import load_wine
from sklearn.utils import Bunch

TARGET = "packages/engine/src/lib/datasets/_data/wine.data.ts"


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
