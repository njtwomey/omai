"""Consistency of decisions among nearest neighbours against AIF360's BinaryLabelDatasetMetric.consistency."""

import aif360
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from whitebox_oracle.fairness import binary_dataset
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"aif360": aif360.__version__}


def cases() -> list[Case]:
    g = rng(66)
    n = 60
    s = (g.uniform(size=n) < 0.5).astype(np.float64)
    x = np.column_stack([g.normal(size=n), g.normal(size=n)])
    y = ((x[:, 0] + 0.3 * g.normal(size=n)) > 0).astype(np.float64)
    data = binary_dataset({"a": x[:, 0], "b": x[:, 1], "s": s}, y, ["s"])
    metric = BinaryLabelDatasetMetric(data, [{"s": 1.0}], [{"s": 0.0}])
    features = np.column_stack([x, s])
    return [
        case(
            f"consistency/k={k}",
            "consistency",
            [features, y],
            np.asarray(float(metric.consistency(n_neighbors=k)[0])),
            k=k,
        )
        for k in [3, 5]
    ]
