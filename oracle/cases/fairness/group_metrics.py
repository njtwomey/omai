"""Group fairness metrics against fairlearn.metrics (MetricFrame and the parity metrics)."""

from typing import Literal

import fairlearn
import fairlearn.metrics as F
import numpy as np
import sklearn.metrics as M
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"fairlearn": fairlearn.__version__}


def ppv_gap(y: FloatArray, p: FloatArray, groups: FloatArray) -> float:
    frame = F.MetricFrame(metrics=M.precision_score, y_true=y, y_pred=p, sensitive_features=groups)
    return float(np.asarray(frame.difference(method="between_groups"), dtype=np.float64))


def treatment(y: FloatArray, p: FloatArray) -> float:
    """FN / FP of one group (NaN without false positives)."""
    _tn, fp, fn, _tp = M.confusion_matrix(y, p, labels=[0, 1]).ravel()
    return float(fn / fp) if fp > 0 else float("nan")


def treatment_gap(y: FloatArray, p: FloatArray, groups: FloatArray) -> float:
    frame = F.MetricFrame(metrics=treatment, y_true=y, y_pred=p, sensitive_features=groups)
    values = np.asarray(frame.by_group, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min())


def scalar(x: object) -> float:
    """A MetricFrame summary (a numpy scalar, or a one-value Series) as a float."""
    return float(np.asarray(x, dtype=np.float64))


def frame_of(y: FloatArray, p: FloatArray, groups: FloatArray, metric: str) -> F.MetricFrame:
    fn = {"accuracy": M.accuracy_score, "recall": M.recall_score}[metric]
    return F.MetricFrame(metrics=fn, y_true=y, y_pred=p, sensitive_features=groups)


def conditional_gaps(p: FloatArray, groups: FloatArray, strata: FloatArray) -> FloatArray:
    """Each stratum's selection-rate gap between groups (MetricFrame with control features), in stratum order."""
    frame = F.MetricFrame(
        metrics=F.selection_rate, y_true=p, y_pred=p, sensitive_features=groups, control_features=strata
    )
    return np.asarray(frame.difference(method="between_groups"), dtype=np.float64)


def balance_gap(y: FloatArray, s: FloatArray, groups: FloatArray, cls: float) -> float:
    def mean_score(yt: FloatArray, sc: FloatArray) -> float:
        return float(np.mean(sc[yt == cls]))

    frame = F.MetricFrame(metrics=mean_score, y_true=y, y_pred=s, sensitive_features=groups)
    return float(np.asarray(frame.difference(), dtype=np.float64))


def cases() -> list[Case]:
    g = rng(64)
    n = 90
    groups = g.integers(0, 3, size=n).astype(np.float64)
    y = (g.uniform(size=n) < 0.4 + 0.1 * groups).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.75, y, 1 - y)
    p = np.where((groups == 2) & (g.uniform(size=n) < 0.2), 1.0, p)
    w = g.uniform(0.3, 2.0, size=n)
    out: list[Case] = []
    methods: list[Literal["between_groups", "to_overall"]] = ["between_groups", "to_overall"]
    for method in methods:
        out += [
            case(
                f"demographicParityDifference/{method}",
                "demographicParityDifference",
                [y, p, groups],
                np.asarray(F.demographic_parity_difference(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
            case(
                f"demographicParityRatio/{method}",
                "demographicParityRatio",
                [y, p, groups],
                np.asarray(F.demographic_parity_ratio(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
            case(
                f"equalOpportunityDifference/{method}",
                "equalOpportunityDifference",
                [y, p, groups],
                np.asarray(F.equal_opportunity_difference(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
            case(
                f"equalOpportunityRatio/{method}",
                "equalOpportunityRatio",
                [y, p, groups],
                np.asarray(F.equal_opportunity_ratio(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
            case(
                f"equalizedOddsDifference/{method}",
                "equalizedOddsDifference",
                [y, p, groups],
                np.asarray(F.equalized_odds_difference(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
            case(
                f"equalizedOddsRatio/{method}",
                "equalizedOddsRatio",
                [y, p, groups],
                np.asarray(F.equalized_odds_ratio(y, p, sensitive_features=groups, method=method)),
                method=method,
            ),
        ]
    out += [
        case(
            "equalizedOddsDifference/mean",
            "equalizedOddsDifference",
            [y, p, groups],
            np.asarray(F.equalized_odds_difference(y, p, sensitive_features=groups, agg="mean")),
            agg="mean",
        ),
        case(
            "demographicParityDifference/weighted",
            "demographicParityDifference",
            [y, p, groups, w],
            np.asarray(F.demographic_parity_difference(y, p, sensitive_features=groups, sample_weight=w)),
        ),
        case(
            "equalizedOddsDifference/weighted",
            "equalizedOddsDifference",
            [y, p, groups, w],
            np.asarray(F.equalized_odds_difference(y, p, sensitive_features=groups, sample_weight=w)),
        ),
        case(
            "predictiveParityDifference",
            "predictiveParityDifference",
            [y, p, groups],
            np.asarray(ppv_gap(y, p, groups)),
        ),
        case(
            "groupRates/selection-0",
            "groupSelection0",
            [y, p, groups],
            np.asarray(float(F.selection_rate(y[groups == 0], p[groups == 0]))),
        ),
        case(
            "groupRates/fpr-1",
            "groupFpr1",
            [y, p, groups],
            np.asarray(float(F.false_positive_rate(y[groups == 1], p[groups == 1]))),
        ),
        case(
            "disparateImpact",
            "disparateImpact",
            [p, groups],
            np.asarray(float(min(p[groups == k].mean() for k in [1, 2]) / p[groups == 0].mean())),
            privileged=0,
        ),
    ]
    strata = g.integers(0, 2, size=n).astype(np.float64)
    scores = np.clip(0.5 * y + 0.15 * groups + g.uniform(size=n) * 0.5, 0, 1)
    for metric in ["accuracy", "recall"]:
        frame = frame_of(y, p, groups, metric)
        out += [
            case(
                f"metricFrame/{metric}/min",
                "frameMin",
                [y, p, groups],
                np.asarray(scalar(frame.group_min())),
                metric=metric,
            ),
            case(
                f"metricFrame/{metric}/max",
                "frameMax",
                [y, p, groups],
                np.asarray(scalar(frame.group_max())),
                metric=metric,
            ),
            case(
                f"metricFrame/{metric}/ratio",
                "frameRatio",
                [y, p, groups],
                np.asarray(scalar(frame.ratio())),
                metric=metric,
            ),
            case(
                f"metricFrame/{metric}/overall",
                "frameOverall",
                [y, p, groups],
                np.asarray(scalar(frame.overall)),
                metric=metric,
            ),
        ]
    accuracy = frame_of(y, p, groups, "accuracy")
    out += [
        case("accuracyDifference", "accuracyDifference", [y, p, groups], np.asarray(scalar(accuracy.difference()))),
        case("accuracyRatio", "accuracyRatio", [y, p, groups], np.asarray(scalar(accuracy.ratio()))),
        case(
            "treatmentEqualityDifference",
            "treatmentEqualityDifference",
            [y, p, groups],
            np.asarray(treatment_gap(y, p, groups)),
        ),
        case(
            "conditionalDemographicParity/strata",
            "conditionalGaps",
            [p, groups, strata],
            conditional_gaps(p, groups, strata),
        ),
        case(
            "balanceForPositiveClass",
            "balanceForPositiveClass",
            [y, scores, groups],
            np.asarray(balance_gap(y, scores, groups, 1.0)),
        ),
        case(
            "balanceForNegativeClass",
            "balanceForNegativeClass",
            [y, scores, groups],
            np.asarray(balance_gap(y, scores, groups, 0.0)),
        ),
    ]
    return out
