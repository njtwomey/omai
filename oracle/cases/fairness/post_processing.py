"""Threshold optimisation against fairlearn's ThresholdOptimizer (each row's probability of a positive decision), and
calibrated equalised odds and reject-option classification against AIF360."""

from typing import Literal, Self

import aif360
import fairlearn
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.base import BaseEstimator, ClassifierMixin

REFERENCE: dict[str, str] = {"fairlearn": fairlearn.__version__, "aif360": aif360.__version__}


class Scores(ClassifierMixin, BaseEstimator):
    """A prefit 'model' whose prediction is its first feature: the scores themselves."""

    def fit(self, x: FloatArray, y: FloatArray) -> Self:
        self.fitted_ = True
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(x, dtype=np.float64)[:, 0]


def decisions(
    scores: FloatArray,
    labels: FloatArray,
    groups: FloatArray,
    constraint: Literal["demographic_parity", "equalized_odds", "true_positive_rate_parity"],
) -> FloatArray:
    """P(positive decision) per row of fairlearn's optimiser fitted on the rows."""
    x = scores.reshape(-1, 1)
    opt = ThresholdOptimizer(
        estimator=Scores().fit(x, labels),
        constraints=constraint,
        objective="accuracy_score",
        prefit=True,
        predict_method="predict",
        grid_size=1000,
    )
    opt.fit(x, labels.astype(np.int64), sensitive_features=groups)
    pmf = np.asarray(opt._pmf_predict(x, sensitive_features=groups), dtype=np.float64)  # pyright: ignore[reportPrivateUsage]
    return pmf[:, 1]


def cases() -> list[Case]:
    g = rng(41)
    n = 160
    groups = (g.uniform(size=n) < 0.35).astype(np.float64)
    latent = g.normal(size=n) - 0.6 * groups
    labels = (latent + 0.8 * g.normal(size=n) > 0).astype(np.float64)
    scores = np.round(1.0 / (1.0 + np.exp(-1.5 * latent)), 6)
    out: list[Case] = []
    names: dict[str, Literal["demographic_parity", "equalized_odds", "true_positive_rate_parity"]] = {
        "demographicParity": "demographic_parity",
        "equalisedOdds": "equalized_odds",
        "truePositiveRateParity": "true_positive_rate_parity",
    }
    for ours, theirs in names.items():
        out.append(
            case(
                ours,
                "decisions",
                [scores, labels, groups],
                decisions(scores, labels, groups, theirs),
                rtol=1e-9,
                atol=1e-9,
                constraint=ours,
            )
        )
    return out + aif360_cases()


def aif360_cases() -> list[Case]:
    """Calibrated equalised odds and reject-option classification against AIF360's post-processors."""
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification
    from mdml_oracle.fairness import binary_dataset, with_labels

    g = rng(87)
    n = 200
    s = (g.uniform(size=n) < 0.4).astype(np.float64)
    latent = g.normal(size=n) - 0.5 * s
    y = (latent + 0.7 * g.normal(size=n) > 0).astype(np.float64)
    scores = np.round(1.0 / (1.0 + np.exp(-1.3 * latent - 0.2 * s)), 6)
    truth = binary_dataset({"s": s}, y, ["s"])
    pred = with_labels(truth, (scores >= 0.5).astype(np.float64))
    pred.scores = scores.reshape(-1, 1)
    out: list[Case] = []
    for cost in ["weighted", "fpr", "fnr"]:
        ceo = CalibratedEqOddsPostprocessing([{"s": 1.0}], [{"s": 0.0}], cost_constraint=cost, seed=0).fit(truth, pred)
        name = {"weighted": "weighted", "fpr": "falsePositive", "fnr": "falseNegative"}[cost]
        out.append(
            case(
                f"calibratedEqualisedOdds/{name}",
                "mixRates",
                [scores, y, s],
                np.asarray([ceo.priv_mix_rate, ceo.unpriv_mix_rate], dtype=np.float64),
                cost=name,
            )
        )
    metrics = {
        "statisticalParity": "Statistical parity difference",
        "averageOdds": "Average odds difference",
        "equalOpportunity": "Equal opportunity difference",
    }
    for ours, theirs in metrics.items():
        roc = RejectOptionClassification(
            [{"s": 1.0}], [{"s": 0.0}], num_class_thresh=40, num_ROC_margin=20, metric_name=theirs
        ).fit(truth, pred)
        out.append(
            case(
                f"rejectOption/{ours}",
                "rejectOption",
                [scores, y, s],
                np.asarray([roc.classification_threshold, roc.ROC_margin], dtype=np.float64),
                metric=ours,
            )
        )
    return out
