"""CART trees against scikit-learn: the grown tree's arrays (features, thresholds, children, impurities, sample counts,
values), the cost-complexity pruning path, the pruned tree and impurity importances, for classification (Gini,
entropy) and regression (squared and absolute error), on continuous data without tied splits."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def arrays(model: DecisionTreeClassifier | DecisionTreeRegressor) -> dict[str, FloatArray]:
    """A fitted tree's arrays as floats (thresholds of leaves set to NaN, as the engine stores them)."""
    t = model.tree_
    leaf = t.children_left < 0
    threshold = np.where(leaf, np.nan, t.threshold)
    value = t.value[:, 0, :]
    value = value / value.sum(axis=1, keepdims=True) if value.shape[1] > 1 else value
    return {
        "feature": np.where(leaf, -1, t.feature).astype(np.float64),
        "threshold": threshold,
        "left": t.children_left.astype(np.float64),
        "right": t.children_right.astype(np.float64),
        "impurity": t.impurity,
        "nSamples": t.n_node_samples.astype(np.float64),
        "value": value,
    }


def cases() -> list[Case]:
    g = rng(41)
    n = 120
    # scikit-learn's trees work on float32 features: start from float32-representable values.
    x = g.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    y = ((x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * g.normal(size=n)) > 0.5).astype(np.float64) + (x[:, 2] > 1.2)
    out: list[Case] = []
    for criterion in ["gini", "entropy"]:
        c = DecisionTreeClassifier(criterion=criterion, max_depth=4, min_samples_leaf=5, random_state=0).fit(x, y)
        path = c.cost_complexity_pruning_path(x, y)
        pruned = DecisionTreeClassifier(
            criterion=criterion, max_depth=4, min_samples_leaf=5, random_state=0, ccp_alpha=float(path.ccp_alphas[-3])
        ).fit(x, y)
        out.append(
            case(
                f"classifier/{criterion}",
                "classifier",
                [x, y],
                {
                    **arrays(c),
                    "alphas": path.ccp_alphas,
                    "pathImpurities": path.impurities,
                    "prunedLeaves": np.array(pruned.get_n_leaves(), dtype=np.float64),
                    "importances": c.feature_importances_,
                    "proba": c.predict_proba(x[:6]),
                },
                rtol=1e-10,
                atol=1e-12,
                criterion=criterion,
                ccpAlpha=float(path.ccp_alphas[-3]),
            )
        )
    t = np.sin(2 * x[:, 0]) + x[:, 1] + 0.2 * g.normal(size=n)
    for criterion in ["squared_error", "absolute_error"]:
        r = DecisionTreeRegressor(criterion=criterion, max_depth=3, min_samples_leaf=5, random_state=0).fit(x, t)
        out.append(
            case(
                f"regressor/{criterion}",
                "regressor",
                [x, t],
                {**arrays(r), "predict": r.predict(x[:6])},
                rtol=1e-10,
                atol=1e-12,
                criterion=criterion,
            )
        )
    return out
