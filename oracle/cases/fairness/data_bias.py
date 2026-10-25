"""Base-rate gaps in a dataset against AIF360's BinaryLabelDatasetMetric, and proxy dependence against sklearn."""

import aif360
import numpy as np
import sklearn.metrics as M
from aif360.metrics import BinaryLabelDatasetMetric
from mdml_oracle.fairness import binary_dataset
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"aif360": aif360.__version__}


def cases() -> list[Case]:
    g = rng(17)
    n = 150
    s = (g.uniform(size=n) < 0.35).astype(np.float64)
    y = (g.uniform(size=n) < 0.6 - 0.25 * s).astype(np.float64)
    metric = BinaryLabelDatasetMetric(binary_dataset({"s": s}, y, ["s"]), [{"s": 1.0}], [{"s": 0.0}])
    proxy = 1.5 * s + g.normal(size=n)
    edges = np.quantile(proxy, np.arange(1, 10) / 10, method="lower")
    binned = np.searchsorted(edges, proxy, side="right").astype(np.float64)
    return [
        case(
            "statisticalParityDifference",
            "statisticalParityDifference",
            [y, s],
            np.asarray(float(metric.statistical_parity_difference())),
        ),
        case("disparateImpact", "disparateImpact", [y, s], np.asarray(float(metric.disparate_impact()))),
        case("baseRate/unprivileged", "baseRate1", [y, s], np.asarray(float(metric.base_rate(privileged=False)))),
        case(
            "proxyDependence",
            "proxyDependence",
            [proxy, s],
            np.asarray(float(M.normalized_mutual_info_score(s, binned))),
        ),
    ]
