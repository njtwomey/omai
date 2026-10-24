"""Shapley explanations against shap: KernelExplainer with every coalition enumerated (exact interventional Shapley
values) on a polynomial regression, TreeExplainer's path-dependent values and interaction values on a decision tree,
and a small random forest. The trees are passed as sklearn's arrays."""

from typing import Any

import numpy as np
import shap
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"shap": shap.__version__, "scikit-learn": sklearn.__version__}


def tree_arrays(est: Any) -> list[FloatArray]:  # noqa: ANN401  (sklearn trees are untyped)
    """children_left, children_right, feature, threshold, value and cover of a fitted sklearn tree."""
    t = est.tree_
    return [
        t.children_left.astype(np.float64),
        t.children_right.astype(np.float64),
        t.feature.astype(np.float64),
        t.threshold.astype(np.float64),
        t.value[:, 0, 0].astype(np.float64),
        t.weighted_n_node_samples.astype(np.float64),
    ]


def cases() -> list[Case]:
    g = rng(4345)
    n, d = 60, 4
    x = g.uniform(-2, 2, size=(n, d))
    y = x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1] - x[:, 2] + 0.3 * g.normal(size=n)
    xe = x[:5]
    background = x[10:30]
    out: list[Case] = []
    model = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())]).fit(x, y)
    explainer: Any = shap.KernelExplainer(model.predict, background)
    values = explainer.shap_values(xe, silent=True)
    out.append(
        case(
            "kernel",
            "kernel",
            [x, y, xe, background],
            {"values": np.asarray(values), "base": np.array(explainer.expected_value)},
            rtol=1e-8,
            atol=1e-10,
        )
    )
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, y)
    te: Any = shap.TreeExplainer(tree)
    out.append(
        case(
            "tree",
            "tree",
            [*tree_arrays(tree), xe],
            {
                "values": np.asarray(te.shap_values(xe)),
                "base": np.array(float(np.ravel(te.expected_value)[0])),
                "interactions": np.asarray(te.shap_interaction_values(xe)).reshape(len(xe), -1),
            },
            rtol=1e-9,
            atol=1e-10,
        )
    )
    forest = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0).fit(x, y)
    fe: Any = shap.TreeExplainer(forest)
    arrays = [a for est in forest.estimators_ for a in tree_arrays(est)]
    out.append(
        case(
            "forest",
            "forest",
            [*arrays, xe],
            {"values": np.asarray(fe.shap_values(xe)), "base": np.array(float(np.ravel(fe.expected_value)[0]))},
            trees=3,
            rtol=1e-9,
            atol=1e-10,
        )
    )
    return out
