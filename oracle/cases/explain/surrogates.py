"""Local and global surrogates: LIME's local linear fit against lime's LimeBase (every feature-selection rule) on a
given neighbourhood, and surrogate CART trees against sklearn's DecisionTreeRegressor and DecisionTreeClassifier."""

from functools import partial
from typing import Any

import lime
import numpy as np
import sklearn
from lime.lime_base import LimeBase
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REFERENCE: dict[str, str] = {"lime": "0.2.0.1", "scikit-learn": sklearn.__version__}
_ = lime


def kernel(d: FloatArray, kernel_width: float) -> FloatArray:
    """lime tabular's kernel."""
    return np.sqrt(np.exp(-(d**2) / kernel_width**2))


def cases() -> list[Case]:
    g = rng(4346)
    m, d = 300, 4
    z = g.normal(size=(m, d))
    z[0] = [0.3, -0.2, 0.5, 0.1]
    labels = np.column_stack([1 / (1 + np.exp(-(2 * z[:, 0] - z[:, 1] + 0.5 * z[:, 0] * z[:, 2])))] * 2)
    labels[:, 1] = 1 - labels[:, 0]
    dist = np.sqrt(((z - z[0]) ** 2).sum(axis=1))
    out: list[Case] = []
    width = 1.5
    base = LimeBase(partial(kernel, kernel_width=width))
    for selection, k in [("none", 4), ("forward_selection", 2), ("highest_weights", 3), ("auto", 7)]:
        result: Any = base.explain_instance_with_data(z, labels, dist, 0, k, feature_selection=selection)
        intercept, pairs, score, local = result
        out.append(
            case(
                f"lime/{selection}",
                "lime",
                [z, labels[:, 0], dist],
                {
                    "intercept": np.array(intercept),
                    "features": np.array([p[0] for p in pairs], dtype=np.float64),
                    "weights": np.array([p[1] for p in pairs]),
                    "score": np.array(score),
                    "local": np.asarray(local),
                },
                selection=selection,
                k=k,
                width=width,
                rtol=1e-8,
                atol=1e-10,
            )
        )
    x = g.uniform(-2, 2, size=(80, 3))
    pred = np.sin(x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2]
    xt = g.uniform(-2, 2, size=(10, 3))
    for depth in [2, 4]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(x, pred)
        out.append(
            case(
                f"tree/regression/{depth}",
                "treeRegression",
                [x, pred, xt],
                {"predict": tree.predict(xt), "train": tree.predict(x)},
                depth=depth,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    labels_c = (pred > np.median(pred)).astype(np.float64) + (x[:, 2] > 1).astype(np.float64)
    tree_c = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(x, labels_c)
    out.append(
        case(
            "tree/classification",
            "treeClassification",
            [x, labels_c, xt],
            {"predict": tree_c.predict(xt), "train": tree_c.predict(x)},
            depth=3,
            minLeaf=3,
        )
    )
    return out
