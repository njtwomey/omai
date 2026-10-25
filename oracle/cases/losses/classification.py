"""Classification losses against torch.nn.functional and torchvision.ops (float64)."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from torchvision.ops import sigmoid_focal_loss

REFERENCE: dict[str, str] = {"torch": torch.__version__, "torchvision": torchvision.__version__}


def categorical_focal(
    logits: FloatArray, labels: FloatArray, gamma: float, alpha: FloatArray | None, reduction: str
) -> FloatArray:
    """-alpha_y (1 - p_y)^gamma log p_y written with torch ops (there is no library function)."""
    logp = F.log_softmax(to_torch(logits), dim=-1)
    idx = torch.tensor(labels, dtype=torch.long)
    picked = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    terms = -((1 - picked.exp()) ** gamma) * picked
    if alpha is not None:
        terms = terms * to_torch(alpha)[idx]
    if reduction == "mean":
        return to_numpy(terms.mean())
    if reduction == "sum":
        return to_numpy(terms.sum())
    return to_numpy(terms)


def cases() -> list[Case]:
    g = rng(40)
    x = g.normal(size=(4, 3))
    logits = g.normal(size=(5, 4)) * 2
    labels = np.array([0.0, 3, 1, 2, 3])
    soft = np.abs(g.normal(size=(5, 4)))
    soft /= soft.sum(-1, keepdims=True)
    weights = np.array([0.5, 1.0, 2.0, 1.5])
    binary = (g.uniform(size=(4, 3)) > 0.5).astype(np.float64)
    probs = g.uniform(0.05, 0.95, size=(4, 3))
    pos_weight = np.array([0.5, 2.0, 1.5])
    long_labels = torch.tensor(labels, dtype=torch.long)
    out: list[Case] = []
    for reduction in ["mean", "sum", "none"]:
        r: dict[str, Any] = {"reduction": reduction}
        out += [
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
                to_numpy(F.cross_entropy(to_torch(logits), long_labels, **r)),
                **r,
            ),
            case(
                f"crossEntropy/soft/{reduction}",
                "crossEntropy",
                [logits, soft],
                to_numpy(F.cross_entropy(to_torch(logits), to_torch(soft), **r)),
                **r,
            ),
            case(
                f"focal/{reduction}",
                "focal",
                [x, binary],
                to_numpy(sigmoid_focal_loss(to_torch(x), to_torch(binary), alpha=0.3, gamma=1.5, reduction=reduction)),
                alpha=0.3,
                gamma=1.5,
                **r,
            ),
            case(
                f"categoricalFocal/{reduction}",
                "categoricalFocal",
                [logits, labels],
                categorical_focal(logits, labels, 2.0, None, reduction),
                **r,
            ),
            case(
                f"multiLabelSoftMargin/{reduction}",
                "multiLabelSoftMargin",
                [x, binary],
                to_numpy(F.multilabel_soft_margin_loss(to_torch(x), to_torch(binary), **r)),
                **r,
            ),
            case(
                f"nllLoss/weighted/{reduction}",
                "nllLoss",
                [np.log(soft), labels, weights],
                to_numpy(F.nll_loss(torch.log(to_torch(soft)), long_labels, weight=to_torch(weights), **r)),
                **r,
            ),
        ]
    out += [
        case(
            "bceWithLogits/posWeight",
            "binaryCrossEntropyWithLogits",
            [x, binary],
            to_numpy(
                F.binary_cross_entropy_with_logits(to_torch(x), to_torch(binary), pos_weight=to_torch(pos_weight))
            ),
            posWeight=pos_weight,
        ),
        case(
            "bceWithLogits/smoothed",
            "binaryCrossEntropyWithLogits",
            [x, binary],
            to_numpy(F.binary_cross_entropy_with_logits(to_torch(x), to_torch(binary * 0.8 + 0.1))),
            labelSmoothing=0.2,
        ),
        case(
            "bce/smoothed",
            "binaryCrossEntropy",
            [probs, binary],
            to_numpy(F.binary_cross_entropy(to_torch(probs), to_torch(binary * 0.9 + 0.05))),
            labelSmoothing=0.1,
        ),
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
            "crossEntropy/soft-smoothed",
            "crossEntropy",
            [logits, soft],
            to_numpy(F.cross_entropy(to_torch(logits), to_torch(soft), label_smoothing=0.3)),
            labelSmoothing=0.3,
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
        case(
            "nllLoss/ignore-weighted",
            "nllLoss",
            [np.log(soft), labels, weights],
            to_numpy(F.nll_loss(torch.log(to_torch(soft)), long_labels, weight=to_torch(weights), ignore_index=1)),
            ignoreIndex=1,
        ),
        case(
            "focal/no-alpha",
            "focal",
            [x, binary],
            to_numpy(sigmoid_focal_loss(to_torch(x), to_torch(binary), alpha=-1, gamma=2, reduction="mean")),
            alpha=-1,
        ),
        case(
            "categoricalFocal/alpha",
            "categoricalFocal",
            [logits, labels, weights],
            categorical_focal(logits, labels, 1.5, weights, "mean"),
            gamma=1.5,
        ),
        case(
            "multiLabelSoftMargin/weighted",
            "multiLabelSoftMargin",
            [x, binary, pos_weight],
            to_numpy(F.multilabel_soft_margin_loss(to_torch(x), to_torch(binary), weight=to_torch(pos_weight))),
        ),
    ]
    return out
