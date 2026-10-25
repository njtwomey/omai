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
    return out
