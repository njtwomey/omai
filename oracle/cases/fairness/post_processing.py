"""Threshold optimisation against fairlearn's ThresholdOptimizer: each row's probability of a positive decision."""

from typing import Literal, Self

import fairlearn
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"fairlearn": fairlearn.__version__}


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
    return out
