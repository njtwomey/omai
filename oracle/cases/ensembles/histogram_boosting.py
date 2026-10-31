"""Histogram gradient boosting against scikit-learn (no early stopping, so deterministic): regression with the squared
error, the Poisson deviance and with missing values and an L2 penalty, and classification with the binary and
multinomial log loss, each with the prediction after every iteration. Features take few enough distinct values for one
bin per value, and a coarser case checks the percentile binning."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def centred(d: FloatArray) -> FloatArray:
    """A stage's raw scores with the per-row mean removed for several classes (the softmax cannot see it)."""
    raw = np.asarray(d, dtype=np.float64).reshape(8, -1)
    return raw - raw.mean(axis=1, keepdims=True) if raw.shape[1] > 1 else raw


def cases() -> list[Case]:
    g = rng(71)
    n = 120
    x = g.normal(size=(n, 3))
    t = np.sin(x[:, 0]) + 0.5 * x[:, 1] + 0.2 * g.normal(size=n)
    out: list[Case] = []
    common = {"max_iter": 12, "learning_rate": 0.2, "max_leaf_nodes": 6, "min_samples_leaf": 5, "early_stopping": False}
    xm = x.copy()
    xm[g.uniform(size=xm.shape) < 0.1] = np.nan
    counts = np.round(np.exp(0.4 * x[:, 0] + 0.3 * x[:, 1]) * 2 + g.uniform(0, 1, size=n), 1)
    for name, xs, y, params in [
        ("squared_error", x, t, {}),
        ("l2", x, t, {"l2_regularization": 2.0}),
        ("missing", xm, t, {}),
        ("poisson", x, counts, {"loss": "poisson"}),
        ("coarse", x, t, {"max_bins": 16}),
    ]:
        r = HistGradientBoostingRegressor(random_state=0, **common, **params).fit(xs, y)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
        out.append(
            case(
                f"regressor/{name}",
                "regressor",
                [xs, y],
                {"predict": r.predict(xs[:8]), "staged": np.array(list(r.staged_predict(xs[:8])))},
                rtol=1e-8,
                atol=1e-10,
                loss=params.get("loss", "squared_error"),
                l2=params.get("l2_regularization", 0.0),
                maxBins=params.get("max_bins", 255),
            )
        )
    for k in [2, 3]:
        y = np.digitize(x[:, 0] + 0.5 * x[:, 2], [0.0] if k == 2 else [-0.5, 0.5]).astype(np.float64)
        c = HistGradientBoostingClassifier(random_state=0, **common).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"classifier/{k}",
                "classifier",
                [x, y],
                {
                    "proba": c.predict_proba(x[:8]),
                    "staged": np.vstack([centred(d) for d in c.staged_decision_function(x[:8])]),
                },
                rtol=1e-8,
                atol=1e-10,
            )
        )
    return out
