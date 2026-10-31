"""AdaBoost (SAMME) against scikit-learn: the member weights and errors, the decision function, the probabilities and
the decision after each stage, for two and three classes, with decision stumps and depth-two trees as the base. The
resampling AdaBoost.R2 draws random bootstraps and is property-tested instead."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def staged(c: AdaBoostClassifier, x: FloatArray) -> FloatArray:
    """The decision function after each stage on some rows, stacked into a matrix with one column per output."""
    rows = [np.asarray(d, dtype=np.float64).reshape(len(x), -1) for d in c.staged_decision_function(x)]
    return np.vstack(rows)


def cases() -> list[Case]:
    g = rng(61)
    n = 90
    # scikit-learn's trees work on float32 features: start from float32-representable values.
    x = g.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    out: list[Case] = []
    for k, depth in [(2, 1), (3, 2)]:
        score = x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * g.normal(size=n)
        y = np.digitize(score, [0.3] if k == 2 else [-0.2, 0.6]).astype(np.float64)
        c = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=8, learning_rate=0.7, random_state=0
        ).fit(x, y)
        out.append(
            case(
                f"classifier/{k}",
                "classifier",
                [x, y],
                {
                    "weights": c.estimator_weights_,
                    "errors": c.estimator_errors_,
                    "decision": np.asarray(c.decision_function(x[:6])).reshape(6, -1),
                    "proba": c.predict_proba(x[:6]),
                    "staged": staged(c, x[:6]),
                    "predict": c.predict(x[:6]),
                },
                rtol=1e-9,
                atol=1e-11,
                nEstimators=8,
                learningRate=0.7,
                maxDepth=depth,
            )
        )
    return out
