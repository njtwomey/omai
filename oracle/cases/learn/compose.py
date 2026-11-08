"""Composition against sklearn.pipeline and sklearn.compose: pipeline predictions and feature names, a pipeline whose
target encoder is cross-fitted in fit, feature unions, column transformers with remainders, and transformed-target
regressors (function pair and transformer)."""

from typing import Any

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler, TargetEncoder

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(4336)
    n = 40
    x = np.column_stack([g.normal(size=n), g.choice([0.0, 1.0, 2.0], size=n), g.uniform(1, 3, size=n)])
    y = 1.5 * x[:, 0] - x[:, 0] ** 2 + 2 * (x[:, 1] == 1) + 0.5 * x[:, 2] + 0.3 * g.normal(size=n)
    z = np.column_stack([g.normal(size=6), g.choice([0.0, 1.0, 2.0], size=6), g.uniform(1, 3, size=6)])
    out: list[Case] = []
    p = Pipeline([("poly", PolynomialFeatures(2)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=0.5))]).fit(x, y)
    names_out: Any = p[:-1].get_feature_names_out()
    names = ",".join(str(s) for s in names_out)
    out.append(
        case(
            "pipeline/poly-ridge",
            "polyRidge",
            [x, y, z],
            {
                "predict": p.predict(z),
                "score": np.array(p.score(x, y)),
                "transform": p[:-1].transform(z),
                "names": names,
            },
            rtol=1e-10,
            atol=1e-10,
        )
    )
    ct = ColumnTransformer(
        [("num", StandardScaler(), [0, 2]), ("cat", OneHotEncoder(sparse_output=False), [1])], remainder="drop"
    )
    tp = Pipeline([("ct", ct), ("lr", LinearRegression())]).fit(x, y)
    out.append(
        case("pipeline/columns-linear", "columnsLinear", [x, y, z], {"predict": tp.predict(z)}, rtol=1e-10, atol=1e-10)
    )
    # sklearn types smooth as str and cv as int; both accept more.
    encoder_options: dict[str, Any] = {"smooth": 1.0, "target_type": "continuous", "cv": KFold(4)}
    te = Pipeline([("te", TargetEncoder(**encoder_options)), ("lr", LinearRegression())])
    xc = x[:, 1:2]
    te.fit(xc, y)
    final: Any = te[-1]
    out.append(
        case(
            "pipeline/target-encoded",
            "targetEncoded",
            [xc, y, z[:, 1:2]],
            {"predict": te.predict(z[:, 1:2]), "coef": final.coef_},
            rtol=1e-10,
            atol=1e-10,
        )
    )
    for remainder in ["drop", "passthrough"]:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(sparse_output=False), [1]), ("mm", MinMaxScaler(), [2])], remainder=remainder
        ).fit(x)
        names = ",".join(str(s).replace(".0", "") for s in ct.get_feature_names_out())
        out.append(
            case(
                f"columns/{remainder}",
                "columns",
                [x, z],
                {"transform": ct.transform(z), "names": names},
                remainder=remainder,
            )
        )
    remainder_options: dict[str, Any] = {"remainder": MinMaxScaler()}
    ct = ColumnTransformer([("num", StandardScaler(), [0])], **remainder_options).fit(x)
    out.append(
        case("columns/remainder-estimator", "columns", [x, z], {"transform": ct.transform(z)}, remainder="minmax")
    )
    fu = FeatureUnion([("z", StandardScaler()), ("m", MinMaxScaler())], transformer_weights={"m": 2.0}).fit(x)
    out.append(case("union", "union", [x, z], {"transform": fu.transform(z)}))
    ypos = np.exp(0.3 * y)
    tt = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp).fit(x, ypos)
    out.append(
        case(
            "target/func",
            "transformedTarget",
            [x, ypos, z],
            {"predict": tt.predict(z)},
            kind="func",
            rtol=1e-10,
            atol=1e-10,
        )
    )
    tt = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler()).fit(x, y)
    out.append(
        case(
            "target/transformer",
            "transformedTarget",
            [x, y, z],
            {"predict": tt.predict(z)},
            kind="transformer",
            rtol=1e-10,
            atol=1e-10,
        )
    )
    return out
