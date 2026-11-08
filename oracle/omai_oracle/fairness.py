"""AIF360 datasets built from arrays, shared by the fairness cases."""

import logging
import warnings

import numpy as np
import pandas as pd

from omai_oracle.helpers import FloatArray

# AIF360 logs a warning for every optional backend it cannot import; keep the generator's output quiet.
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from aif360.datasets import BinaryLabelDataset  # noqa: E402


def binary_dataset(features: dict[str, FloatArray], labels: FloatArray, protected: list[str]) -> BinaryLabelDataset:
    """A BinaryLabelDataset of named feature columns, labels (1 favourable) and the protected columns' names."""
    frame = pd.DataFrame({**{k: np.asarray(v, dtype=np.float64) for k, v in features.items()}, "y": labels})
    return BinaryLabelDataset(
        df=frame, label_names=["y"], protected_attribute_names=protected, favorable_label=1.0, unfavorable_label=0.0
    )


def with_labels(dataset: BinaryLabelDataset, labels: FloatArray) -> BinaryLabelDataset:
    """A copy of a dataset with other labels (a classifier's predictions)."""
    out = dataset.copy(deepcopy=True)
    out.labels = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    return out
