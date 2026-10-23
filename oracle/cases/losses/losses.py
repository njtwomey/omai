"""Losses against torch.nn.functional (float64)."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(40)
    x, y = g.normal(size=(4, 3)), g.normal(size=(4, 3))
    var = np.exp(g.normal(size=(4, 3)))
    counts = g.poisson(2.0, size=(4, 3)).astype(np.float64)
    logits = g.normal(size=(5, 4)) * 2
    labels = np.array([0.0, 3, 1, 2, 3])
    soft = np.abs(g.normal(size=(5, 4)))
    soft /= soft.sum(-1, keepdims=True)
    weights = np.array([0.5, 1.0, 2.0, 1.5])
    binary = (g.uniform(size=(4, 3)) > 0.5).astype(np.float64)
    probs = g.uniform(0.05, 0.95, size=(4, 3))
    out: list[Case] = []
    for reduction in ["mean", "sum", "none"]:
        r: dict[str, Any] = {"reduction": reduction}
        out += [
            case(f"mse/{reduction}", "mse", [x, y], to_numpy(F.mse_loss(to_torch(x), to_torch(y), **r)), **r),
            case(f"mae/{reduction}", "mae", [x, y], to_numpy(F.l1_loss(to_torch(x), to_torch(y), **r)), **r),
            case(
                f"huber/{reduction}",
                "huber",
                [x, y],
                to_numpy(F.huber_loss(to_torch(x), to_torch(y), delta=0.7, **r)),
                delta=0.7,
                **r,
            ),
            case(
                f"poissonNll/{reduction}",
                "poissonNll",
                [x, counts],
                to_numpy(F.poisson_nll_loss(to_torch(x), to_torch(counts), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"gaussianNll/{reduction}",
                "gaussianNll",
                [x, y, var],
                to_numpy(F.gaussian_nll_loss(to_torch(x), to_torch(y), to_torch(var), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"bceWithLogits/{reduction}",
                "binaryCrossEntropyWithLogits",
                [x, binary],
                to_numpy(F.binary_cross_entropy_with_logits(to_torch(x), to_torch(binary), **r)),
                **r,
            ),
            case(
                f"bce/{reduction}",
                "binaryCrossEntropy",
                [probs, binary],
                to_numpy(F.binary_cross_entropy(to_torch(probs), to_torch(binary), **r)),
                **r,
            ),
            case(
                f"crossEntropy/{reduction}",
                "crossEntropy",
                [logits, labels],
                to_numpy(F.cross_entropy(to_torch(logits), torch.tensor(labels, dtype=torch.long), **r)),
                **r,
            ),
            case(
                f"crossEntropy/soft/{reduction}",
                "crossEntropy",
                [logits, soft],
                to_numpy(F.cross_entropy(to_torch(logits), to_torch(soft), **r)),
                **r,
            ),
        ]
    long_labels = torch.tensor(labels, dtype=torch.long)
    out += [
        case(
            "crossEntropy/weighted",
            "crossEntropy",
            [logits, labels, weights],
            to_numpy(F.cross_entropy(to_torch(logits), long_labels, weight=to_torch(weights))),
        ),
        case(
            "crossEntropy/smoothed-weighted",
            "crossEntropy",
            [logits, labels, weights],
            to_numpy(F.cross_entropy(to_torch(logits), long_labels, weight=to_torch(weights), label_smoothing=0.2)),
            labelSmoothing=0.2,
        ),
        case(
            "crossEntropy/ignore",
            "crossEntropy",
            [logits, labels],
            to_numpy(F.cross_entropy(to_torch(logits), long_labels, ignore_index=3)),
            ignoreIndex=3,
        ),
        case(
            "nllLoss",
            "nllLoss",
            [np.log(soft), labels],
            to_numpy(F.nll_loss(torch.log(to_torch(soft)), long_labels)),
        ),
    ]
    return out
