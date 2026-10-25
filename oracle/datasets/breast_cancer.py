"""The Wisconsin diagnostic breast cancer data, from scikit-learn."""

from typing import cast

import sklearn
from mdml_oracle.vendor import render_classification
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

TARGET = "packages/engine/src/lib/datasets/_data/breast-cancer.data.ts"


def render() -> str:
    bunch = cast(Bunch, load_breast_cancer())
    return render_classification(
        "BREAST_CANCER",
        "breast_cancer",
        f"scikit-learn {sklearn.__version__} (load_breast_cancer)",
        bunch.data,
        bunch.target,
        list(bunch.feature_names),
        list(bunch.target_names),
    )
