"""Padding and class-balanced weights against torch and scikit-learn."""

import numpy as np
import sklearn
import torch
from mdml_oracle.fixture import Case, case
from sklearn.utils.class_weight import compute_sample_weight
from torch.nn.utils.rnn import pad_sequence

REFERENCE: dict[str, str] = {"torch": torch.__version__, "scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    seqs = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]]
    padded = pad_sequence([torch.tensor(s, dtype=torch.float64) for s in seqs], batch_first=True, padding_value=-1.0)
    out.append(case("padSequences/post", "padSequences", [seqs], padded.numpy(), padValue=-1.0))
    left = pad_sequence(
        [torch.tensor(s, dtype=torch.float64) for s in seqs], batch_first=True, padding_value=0.0, padding_side="left"
    )
    out.append(case("padSequences/pre", "padSequences", [seqs], left.numpy(), padValue=0.0, padding="pre"))
    vectors = [np.arange(6, dtype=np.float64).reshape(3, 2), np.arange(2, dtype=np.float64).reshape(1, 2) + 10]
    stacked = pad_sequence([torch.tensor(v) for v in vectors], batch_first=True)
    out.append(case("padSequences/vectors", "padSequences", vectors, stacked.numpy(), padValue=0.0))
    for name, y in [("binary", [0, 0, 0, 1]), ("three", [2, 0, 1, 1, 2, 2, 2])]:
        w = compute_sample_weight("balanced", np.asarray(y))
        out.append(case(f"balancedWeights/{name}", "balancedWeights", [np.asarray(y)], np.asarray(w)))
    return out
