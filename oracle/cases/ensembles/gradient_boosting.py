"""Gradient boosting against scikit-learn (no subsampling, so deterministic): regression with the squared, Huber
and quantile losses, and classification with the binary and multinomial log loss and the exponential loss,
each with the raw prediction after every stage (centred over the classes, which the softmax cannot see)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def centred(d: FloatArray) -> FloatArray:
    """A stage's raw scores with the per-row mean removed for several classes (the softmax cannot see it)."""
    raw = np.asarray(d, dtype=np.float64).reshape(8, -1)
    return raw - raw.mean(axis=1, keepdims=True) if raw.shape[1] > 1 else raw


def cases() -> list[Case]:
    g = rng(51)
    n = 80
    # scikit-learn's trees work on float32 features: start from float32-representable values.
    x = g.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    t = np.sin(x[:, 0]) + 0.5 * x[:, 1] + 0.2 * g.normal(size=n)
    out: list[Case] = []
    # The absolute error is left out: its sign gradients make many splits tie exactly, and the leaf medians then
    # depend on which tied split scikit-learn meets first.
    for loss in ["squared_error", "huber", "quantile"]:
        # Sign gradients tie across features, broken by scikit-learn's random feature order: one feature for those.
        xs: FloatArray = x if loss in ("squared_error", "huber") else x[:, :1]
        r = GradientBoostingRegressor(
            loss=loss,  # pyright: ignore[reportArgumentType]
            n_estimators=15,
            max_depth=2,
            learning_rate=0.3,
            alpha=0.9,
            random_state=0,
        ).fit(xs, t)
        out.append(
            case(
                f"regressor/{loss}",
                "regressor",
                [xs, t],
                {
                    "predict": r.predict(xs[:8]),
                    "staged": np.array([p for p in r.staged_predict(xs[:8])]),
                },
                rtol=1e-9,
                atol=1e-10,
                loss=loss,
                alpha=0.9,
            )
        )
    for k in [2, 3]:
        y = np.digitize(x[:, 0] + 0.5 * x[:, 2], [0.0] if k == 2 else [-0.5, 0.5]).astype(np.float64)
        c = GradientBoostingClassifier(n_estimators=10, max_depth=2, learning_rate=0.5, random_state=0).fit(x, y)
        out.append(
            case(
                f"classifier/{k}",
                "classifier",
                [x, y],
                {
                    "proba": c.predict_proba(x[:8]),
                    "staged": np.vstack([centred(d) for d in c.staged_decision_function(x[:8])]),
                },
                rtol=1e-9,
                atol=1e-10,
                loss="log_loss",
            )
        )
    y2 = np.digitize(x[:, 0] + 0.5 * x[:, 2], [0.0]).astype(np.float64)
    e = GradientBoostingClassifier(loss="exponential", n_estimators=10, max_depth=2, learning_rate=0.5, random_state=0)
    e.fit(x, y2)
    out.append(
        case(
            "classifier/exponential",
            "classifier",
            [x, y2],
            {"proba": e.predict_proba(x[:8]), "decision": np.asarray(e.decision_function(x[:8])).reshape(8, -1)},
            rtol=1e-9,
            atol=1e-10,
            loss="exponential",
        )
    )
    return out
