"""k-nearest neighbours against scikit-learn: classifier probabilities and regressor predictions with uniform and
distance weights, and Manhattan distances."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(61)
    x = g.normal(size=(60, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float64) + (x[:, 2] > 1)
    t = x[:, 0] - 2 * x[:, 1] + 0.1 * g.normal(size=60)
    q = g.normal(size=(7, 3))
    out: list[Case] = []
    for weights in ["uniform", "distance"]:
        for metric in ["euclidean", "manhattan"]:
            c = KNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric).fit(x, y)  # pyright: ignore[reportArgumentType]
            r = KNeighborsRegressor(n_neighbors=4, weights=weights, metric=metric).fit(x, t)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"knn/{weights}/{metric}",
                    "knn",
                    [x, y, t, q],
                    {"proba": c.predict_proba(q), "predict": r.predict(q)},
                    weights=weights,
                    metric=metric,
                )
            )
    rr = RadiusNeighborsRegressor(radius=1.5).fit(x, t)
    out.append(case("radius", "radius", [x, t, q], {"predict": rr.predict(q)}, radius=1.5))
    return out
