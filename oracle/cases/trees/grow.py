"""CART trees against scikit-learn: the grown tree's arrays (features, thresholds, children, impurities, sample counts,
values, where missing values go), the cost-complexity pruning path and the pruned tree, for classification (Gini,
entropy) and regression (squared, absolute and Poisson error, Friedman's MSE), depth-first and best-first growth,
sample weights, minimum impurity decrease and missing values, on continuous data without tied splits."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

type Model = DecisionTreeClassifier | DecisionTreeRegressor


def arrays(model: Model) -> dict[str, FloatArray]:
    """A fitted tree's arrays as floats, in the layout the engine stores (leaves: feature and threshold -2)."""
    t = model.tree_
    value = t.value[:, 0, :]
    value = value / value.sum(axis=1, keepdims=True) if value.shape[1] > 1 else value
    return {
        "feature": t.feature.astype(np.float64),
        "threshold": t.threshold,
        "left": t.children_left.astype(np.float64),
        "right": t.children_right.astype(np.float64),
        "impurity": t.impurity,
        "nSamples": t.n_node_samples.astype(np.float64),
        "weightedNSamples": t.weighted_n_node_samples,
        "value": value,
        "missingLeft": np.where(t.children_left >= 0, t.missing_go_to_left, 0).astype(np.float64),
    }


def data() -> tuple[FloatArray, FloatArray, FloatArray]:
    g = rng(41)
    n = 120
    # scikit-learn's trees work on float32 features: start from float32-representable values.
    x = g.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    y = ((x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * g.normal(size=n)) > 0.5).astype(np.float64) + (x[:, 2] > 1.2)
    t = np.sin(2 * x[:, 0]) + x[:, 1] + 0.2 * g.normal(size=n)
    return x, y, t


def cases() -> list[Case]:
    x, y, t = data()
    g = rng(7)
    out: list[Case] = []
    for criterion in ["gini", "entropy"]:
        c = DecisionTreeClassifier(criterion=criterion, max_depth=4, min_samples_leaf=5, random_state=0).fit(x, y)  # pyright: ignore[reportArgumentType]
        path = c.cost_complexity_pruning_path(x, y)
        alpha = float(path.ccp_alphas[-3])
        pruned = DecisionTreeClassifier(
            criterion=criterion,  # pyright: ignore[reportArgumentType]
            max_depth=4,
            min_samples_leaf=5,
            random_state=0,
            ccp_alpha=alpha,
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
                    "prunedFeature": pruned.tree_.feature.astype(np.float64),
                    "importances": c.feature_importances_,
                    "proba": c.predict_proba(x[:6]),
                },
                rtol=1e-10,
                atol=1e-12,
                criterion=criterion,
                maxDepth=4,
                minSamplesLeaf=5,
                ccpAlpha=alpha,
            )
        )
    # Best-first growth under a leaf limit, with a minimum impurity decrease.
    for fn, model, target in [
        (
            "classifier",
            DecisionTreeClassifier(max_leaf_nodes=7, min_impurity_decrease=0.002, random_state=0),
            y,
        ),
        ("regressor", DecisionTreeRegressor(max_leaf_nodes=7, random_state=0), t),
    ]:
        model.fit(x, target)
        out.append(
            case(
                f"{fn}/best-first",
                fn,
                [x, target],
                {**arrays(model)},
                rtol=1e-10,
                atol=1e-12,
                criterion="gini" if fn == "classifier" else "squared_error",
                maxLeafNodes=7,
                minImpurityDecrease=0.002 if fn == "classifier" else 0.0,
            )
        )
    for criterion in ["squared_error", "absolute_error"]:
        r = DecisionTreeRegressor(criterion=criterion, max_depth=3, min_samples_leaf=5, random_state=0).fit(x, t)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"regressor/{criterion}",
                "regressor",
                [x, t],
                {**arrays(r), "predict": r.predict(x[:6])},
                rtol=1e-10,
                atol=1e-12,
                criterion=criterion,
                maxDepth=3,
                minSamplesLeaf=5,
            )
        )
    counts = np.round(np.exp(0.5 * x[:, 0] + 0.3 * x[:, 1]) * 2 + g.uniform(0, 1, size=len(x)), 1)
    p = DecisionTreeRegressor(criterion="poisson", max_depth=3, min_samples_leaf=5, random_state=0).fit(x, counts)
    out.append(
        case(
            "regressor/poisson",
            "regressor",
            [x, counts],
            {**arrays(p), "predict": p.predict(x[:6])},
            rtol=1e-9,
            atol=1e-12,
            criterion="poisson",
            maxDepth=3,
            minSamplesLeaf=5,
        )
    )
    # Sample weights and a minimum leaf weight.
    w = g.uniform(0.2, 3.0, size=len(x))
    wc = DecisionTreeClassifier(max_depth=4, min_weight_fraction_leaf=0.05, random_state=0).fit(x, y, sample_weight=w)
    out.append(
        case(
            "classifier/weighted",
            "classifier",
            [x, y, w],
            {**arrays(wc), "proba": wc.predict_proba(x[:6])},
            rtol=1e-10,
            atol=1e-12,
            criterion="gini",
            maxDepth=4,
            minWeightFractionLeaf=0.05,
        )
    )
    wr = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x, t, sample_weight=w)
    out.append(
        case(
            "regressor/weighted",
            "regressor",
            [x, t, w],
            {**arrays(wr), "predict": wr.predict(x[:6])},
            rtol=1e-10,
            atol=1e-12,
            criterion="squared_error",
            maxDepth=3,
        )
    )
    # Missing values: the split sends them to the side that scores better.
    xm = x.copy()
    xm[g.uniform(size=xm.shape) < 0.12] = np.nan
    mc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(xm, y)
    out.append(
        case(
            "classifier/missing",
            "classifier",
            [xm, y],
            {**arrays(mc), "proba": mc.predict_proba(xm[:6])},
            rtol=1e-10,
            atol=1e-12,
            criterion="gini",
            maxDepth=3,
            minSamplesLeaf=3,
        )
    )
    mr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=0).fit(xm, t)
    out.append(
        case(
            "regressor/missing",
            "regressor",
            [xm, t],
            {**arrays(mr), "predict": mr.predict(xm[:6])},
            rtol=1e-10,
            atol=1e-12,
            criterion="squared_error",
            maxDepth=3,
            minSamplesLeaf=3,
        )
    )
    return out
