"""Cross-validation against sklearn.model_selection: cross_validate (several scorers, train scores, stratified folds
for classifiers), cross_val_predict (values and aligned probabilities), learning and validation curves."""

from typing import Any

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    validation_curve,
)
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(4337)
    n = 36
    x = g.normal(size=(n, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.5 * g.normal(size=n)
    yc = (x[:, 0] + 0.8 * x[:, 1] + 0.7 * g.normal(size=n) > 0).astype(np.float64)
    y3 = np.digitize(x[:, 0] - x[:, 2] + 0.5 * g.normal(size=n), [-0.7, 0.7]).astype(np.float64)
    out: list[Case] = []
    r: dict[str, Any] = cross_validate(
        Ridge(alpha=1.0), x, y, cv=KFold(4), scoring=["r2", "neg_mean_squared_error"], return_train_score=True
    )
    out.append(
        case(
            "crossValidate/ridge",
            "crossValidateRidge",
            [x, y],
            {
                "testR2": r["test_r2"],
                "trainR2": r["train_r2"],
                "testMse": r["test_neg_mean_squared_error"],
                "trainMse": r["train_neg_mean_squared_error"],
            },
            rtol=1e-10,
            atol=1e-12,
        )
    )
    r = cross_validate(GaussianNB(), x, yc, cv=3, scoring=["accuracy", "neg_log_loss", "roc_auc", "neg_brier_score"])
    out.append(
        case(
            "crossValidate/nb",
            "crossValidateNb",
            [x, yc],
            {
                "accuracy": r["test_accuracy"],
                "logLoss": r["test_neg_log_loss"],
                "rocAuc": r["test_roc_auc"],
                "brier": r["test_neg_brier_score"],
            },
            rtol=1e-9,
            atol=1e-12,
        )
    )
    out.append(
        case(
            "crossValPredict/ridge",
            "crossValPredictRidge",
            [x, y],
            {"predict": cross_val_predict(Ridge(alpha=1.0), x, y, cv=KFold(5))},
            rtol=1e-10,
            atol=1e-12,
        )
    )
    out.append(
        case(
            "crossValPredict/proba",
            "crossValPredictProba",
            [x, y3],
            {"proba": cross_val_predict(GaussianNB(), x, y3, cv=StratifiedKFold(4), method="predict_proba")},
            rtol=1e-9,
            atol=1e-12,
        )
    )
    sizes, train, test = learning_curve(
        Ridge(alpha=1.0), x, y, train_sizes=np.array([0.2, 0.5, 1.0]), cv=KFold(4), scoring="r2"
    )[:3]
    out.append(
        case(
            "learningCurve",
            "learningCurve",
            [x, y],
            {"sizes": np.asarray(sizes, dtype=np.int32), "train": train, "test": test},
            rtol=1e-10,
            atol=1e-12,
        )
    )
    train, test = validation_curve(
        Ridge(),
        x,
        y,
        param_name="alpha",
        param_range=[0.01, 1.0, 100.0],
        cv=KFold(3),
        scoring="neg_mean_absolute_error",
    )
    out.append(
        case("validationCurve", "validationCurve", [x, y], {"train": train, "test": test}, rtol=1e-10, atol=1e-12)
    )
    return out
