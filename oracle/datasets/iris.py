"""Fisher's iris data, from scikit-learn."""

from typing import cast

import sklearn
from omai_oracle.vendor import render_classification
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

TARGET = "packages/engine/src/lib/datasets/_data/iris.data.ts"


def render() -> str:
    bunch = cast(Bunch, load_iris())
    return render_classification(
        "IRIS",
        "iris",
        f"scikit-learn {sklearn.__version__} (load_iris)",
        bunch.data,
        bunch.target,
        list(bunch.feature_names),
        list(bunch.target_names),
    )
