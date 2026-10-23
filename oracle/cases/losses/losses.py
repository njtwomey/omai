"""Losses against torch.nn.functional (float64)."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def t(x: FloatArray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def value(x: torch.Tensor) -> FloatArray:
    return x.detach().numpy()


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
            case(f"mse/{reduction}", "mse", [x, y], value(F.mse_loss(t(x), t(y), **r)), **r),
            case(f"mae/{reduction}", "mae", [x, y], value(F.l1_loss(t(x), t(y), **r)), **r),
            case(
                f"huber/{reduction}", "huber", [x, y], value(F.huber_loss(t(x), t(y), delta=0.7, **r)), delta=0.7, **r
            ),
            case(
                f"poissonNll/{reduction}",
                "poissonNll",
                [x, counts],
                value(F.poisson_nll_loss(t(x), t(counts), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"gaussianNll/{reduction}",
                "gaussianNll",
                [x, y, var],
                value(F.gaussian_nll_loss(t(x), t(y), t(var), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"bceWithLogits/{reduction}",
                "binaryCrossEntropyWithLogits",
                [x, binary],
                value(F.binary_cross_entropy_with_logits(t(x), t(binary), **r)),
                **r,
            ),
            case(
                f"bce/{reduction}",
                "binaryCrossEntropy",
                [probs, binary],
                value(F.binary_cross_entropy(t(probs), t(binary), **r)),
                **r,
            ),
            case(
                f"crossEntropy/{reduction}",
                "crossEntropy",
                [logits, labels],
                value(F.cross_entropy(t(logits), torch.tensor(labels, dtype=torch.long), **r)),
                **r,
            ),
            case(
                f"crossEntropy/soft/{reduction}",
                "crossEntropy",
                [logits, soft],
                value(F.cross_entropy(t(logits), t(soft), **r)),
                **r,
            ),
        ]
    long_labels = torch.tensor(labels, dtype=torch.long)
    out += [
        case(
            "crossEntropy/weighted",
            "crossEntropy",
            [logits, labels, weights],
            value(F.cross_entropy(t(logits), long_labels, weight=t(weights))),
        ),
        case(
            "crossEntropy/smoothed-weighted",
            "crossEntropy",
            [logits, labels, weights],
            value(F.cross_entropy(t(logits), long_labels, weight=t(weights), label_smoothing=0.2)),
            labelSmoothing=0.2,
        ),
        case(
            "crossEntropy/ignore",
            "crossEntropy",
            [logits, labels],
            value(F.cross_entropy(t(logits), long_labels, ignore_index=3)),
            ignoreIndex=3,
        ),
        case(
            "nllLoss",
            "nllLoss",
            [np.log(soft), labels],
            value(F.nll_loss(torch.log(t(soft)), long_labels)),
        ),
    ]
    return out
