"""Fairness of regressors and scores against scipy (Kolmogorov-Smirnov and Wasserstein distances between groups) and
fairlearn's MetricFrame (each group's mean loss)."""

import fairlearn.metrics as F
import numpy as np
import scipy
import scipy.stats as S
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(31)
    n = 90
    groups = (g.uniform(size=n) < 0.4).astype(np.float64)
    y = g.normal(size=n) + 0.5 * groups
    pred = y + g.normal(size=n) * (0.5 + 0.5 * groups) - 0.3 * groups
    scores = 1 / (1 + np.exp(-(pred - 0.2)))
    labels = (y > 0).astype(np.float64)
    a, b = scores[groups == 0], scores[groups == 1]
    frame = F.MetricFrame(metrics=M.mean_squared_error, y_true=y, y_pred=pred, sensitive_features=groups)
    pos = labels == 1
    return [
        case("distributionParity/ks", "ks", [scores, groups], np.asarray(float(S.ks_2samp(a, b).statistic))),
        case(
            "distributionParity/wasserstein",
            "wasserstein",
            [scores, groups],
            np.asarray(float(S.wasserstein_distance(a, b))),
        ),
        case(
            "equalOpportunityInDistribution/ks",
            "eoKs",
            [scores, labels, groups],
            np.asarray(float(S.ks_2samp(scores[pos & (groups == 0)], scores[pos & (groups == 1)]).statistic)),
        ),
        case(
            "boundedGroupLoss/squared",
            "groupLoss",
            [y, pred, groups],
            np.asarray(frame.by_group, dtype=np.float64),
        ),
    ]
