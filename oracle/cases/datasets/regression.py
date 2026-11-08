"""Regression dataset generators and real datasets against sklearn.datasets (truth functions on sklearn's samples)."""

from typing import cast

import sklearn
from omai_oracle.fixture import Case, case
from sklearn.datasets import load_diabetes, make_friedman1, make_friedman2, make_friedman3
from sklearn.utils import Bunch

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    # Noise-free samples, so sklearn's targets are the regression function at its own features.
    for name, make in [("friedman1", make_friedman1), ("friedman2", make_friedman2), ("friedman3", make_friedman3)]:
        x, y = make(n_samples=30, noise=0.0, random_state=0)
        out.append(case(f"{name}/regression-function", f"{name}Function", [x], y, rtol=1e-12))
    # load_diabetes returns a Bunch unless return_X_y is set.
    scaled = cast(Bunch, load_diabetes())
    raw = cast(Bunch, load_diabetes(scaled=False))
    out.append(case("diabetes/raw-x", "diabetes", [], raw.data, rtol=0, atol=0, part="x", scaled=False))
    out.append(case("diabetes/y", "diabetes", [], raw.target, rtol=0, atol=0, part="y", scaled=False))
    out.append(case("diabetes/scaled-x", "diabetes", [], scaled.data, rtol=1e-12, atol=1e-15, part="x", scaled=True))
    return out
