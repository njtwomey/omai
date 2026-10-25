"""fairlearn's ExponentiatedGradient and GridSearch with a deterministic learner (a weighted least-squares classifier),
so the reductions themselves are compared: the final mixture's weights, its decisions, and the grid's points."""

from typing import Self

import fairlearn
import numpy as np
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient, GridSearch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"fairlearn": fairlearn.__version__}


class LeastSquaresClassifier(ClassifierMixin, BaseEstimator):
    """Weighted least squares on the 0/1 target, predicting 1 where the fit is at least 0.5."""

    def fit(self, x: FloatArray, y: FloatArray, sample_weight: FloatArray | None = None) -> Self:
        self.model_ = LinearRegression().fit(x, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        return (np.asarray(self.model_.predict(x)) >= 0.5).astype(np.float64)


def data() -> tuple[FloatArray, FloatArray, FloatArray]:
    g = rng(29)
    n = 120
    groups = (g.uniform(size=n) < 0.4).astype(np.float64)
    x = np.column_stack([g.normal(size=n) - 0.8 * groups, g.normal(size=n)])
    y = (x[:, 0] + 0.5 * x[:, 1] + 0.6 * g.normal(size=n) > -0.2).astype(np.float64)
    return x, y, groups


def cases() -> list[Case]:
    x, y, groups = data()
    out: list[Case] = []
    for name, moment in [("demographicParity", DemographicParity), ("equalisedOdds", EqualizedOdds)]:
        eg = ExponentiatedGradient(LeastSquaresClassifier(), moment(), eps=0.02)
        eg.fit(x, y, sensitive_features=groups)
        pmf = np.asarray(eg._pmf_predict(x), dtype=np.float64)[:, 1]  # pyright: ignore[reportPrivateUsage]
        weights = np.asarray(eg.weights_.sort_index(), dtype=np.float64)
        gs = GridSearch(LeastSquaresClassifier(), moment(), grid_size=11)
        gs.fit(x, y, sensitive_features=groups)
        out += [
            case(
                f"exponentiatedGradient/{name}/weights",
                "egWeights",
                [x, y, groups],
                weights,
                rtol=1e-8,
                atol=1e-8,
                constraint=name,
            ),
            case(
                f"exponentiatedGradient/{name}/decisions",
                "egDecisions",
                [x, y, groups],
                pmf,
                rtol=1e-8,
                atol=1e-8,
                constraint=name,
            ),
            case(
                f"exponentiatedGradient/{name}/bestIteration",
                "egBestIteration",
                [x, y, groups],
                np.asarray(float(eg.best_iter_ + 1)),
                constraint=name,
            ),
            case(
                f"gridSearch/{name}/errors",
                "gridErrors",
                [x, y, groups],
                np.asarray(gs.objectives_, dtype=np.float64),
                rtol=1e-10,
                atol=1e-10,
                constraint=name,
            ),
            case(
                f"gridSearch/{name}/violations",
                "gridViolations",
                [x, y, groups],
                np.asarray(gs.gammas_.max(axis=0), dtype=np.float64),
                rtol=1e-10,
                atol=1e-10,
                constraint=name,
            ),
            case(
                f"gridSearch/{name}/best", "gridBest", [x, y, groups], np.asarray(float(gs.best_idx_)), constraint=name
            ),
        ]
    return out
