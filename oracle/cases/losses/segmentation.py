"""Segmentation losses against MONAI (channels moved last), and the Lovasz losses of Berman et al. in numpy.

The Lovasz hinge and Lovasz-softmax references are a direct numpy port of the authors' PyTorch code
(github.com/bermanmaxim/LovaszSoftmax); focal Tversky is (1 - TI)^gamma on MONAI's Tversky index.
"""

from typing import Any

import monai
import numpy as np
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from monai.losses.dice import DiceLoss
from monai.losses.tversky import TverskyLoss

REFERENCE: dict[str, str] = {"monai": monai.__version__, "torch": torch.__version__}


def channels_first(x: FloatArray) -> torch.Tensor:
    """[N, ..., C] to MONAI's [N, C, ...]."""
    return to_torch(np.moveaxis(x, -1, 1))


def labels_first(y: FloatArray) -> torch.Tensor:
    """Index targets [N, ...] to MONAI's [N, 1, ...]."""
    return to_torch(y[:, None])


def monai_none(loss: torch.Tensor) -> FloatArray:
    """MONAI's per-sample-and-class output [N, C, 1, ...] squeezed to [N, C] (or [C] when pooled)."""
    return to_numpy(loss).reshape(loss.shape[0], -1) if loss.ndim > 1 else to_numpy(loss)


def lovasz_grad(gt_sorted: FloatArray) -> FloatArray:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum()
    union = gts + (1 - gt_sorted).cumsum()
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: FloatArray, labels: FloatArray) -> float:
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    perm = np.argsort(-errors, kind="stable")
    return float(np.dot(np.maximum(errors[perm], 0), lovasz_grad(labels[perm])))


def lovasz_softmax_flat(probs: FloatArray, labels: FloatArray, classes: str) -> float:
    losses: list[float] = []
    for c in range(probs.shape[1]):
        fg = (labels == c).astype(np.float64)
        if classes == "present" and fg.sum() == 0:
            continue
        errors = np.abs(fg - probs[:, c])
        perm = np.argsort(-errors, kind="stable")
        losses.append(float(np.dot(errors[perm], lovasz_grad(fg[perm]))))
    return float(np.mean(losses))


def cases() -> list[Case]:
    g = rng(46)
    n, h, w, c = 2, 4, 5, 3
    logits = g.normal(size=(n, h, w, c)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    labels = g.integers(0, c, size=(n, h, w)).astype(np.float64)
    onehot = np.eye(c)[labels.astype(np.int64)]
    weights = np.array([0.5, 1.0, 2.0])
    out: list[Case] = []
    for reduction in ["mean", "sum", "none"]:
        r: dict[str, Any] = {"reduction": reduction}
        dl = DiceLoss(to_onehot_y=True, reduction=reduction)
        tl = TverskyLoss(to_onehot_y=True, alpha=0.3, beta=0.7, reduction=reduction)
        out += [
            case(
                f"dice/{reduction}",
                "dice",
                [probs, labels],
                monai_none(dl(channels_first(probs), labels_first(labels))),
                **r,
            ),
            case(
                f"tversky/{reduction}",
                "tversky",
                [probs, labels],
                monai_none(tl(channels_first(probs), labels_first(labels))),
                alpha=0.3,
                beta=0.7,
                **r,
            ),
        ]
    out += [
        case(
            "dice/softmax-logits",
            "dice",
            [logits, labels],
            to_numpy(DiceLoss(to_onehot_y=True, softmax=True)(channels_first(logits), labels_first(labels))),
            activation="softmax",
        ),
        case(
            "dice/sigmoid-onehot-squared",
            "dice",
            [logits, onehot],
            to_numpy(DiceLoss(sigmoid=True, squared_pred=True)(channels_first(logits), channels_first(onehot))),
            activation="sigmoid",
            squared=True,
        ),
        case(
            "dice/no-background-batch",
            "dice",
            [probs, labels],
            to_numpy(
                DiceLoss(to_onehot_y=True, include_background=False, batch=True)(
                    channels_first(probs), labels_first(labels)
                )
            ),
            includeBackground=False,
            batch=True,
        ),
        case(
            "dice/smooth",
            "dice",
            [probs, labels],
            to_numpy(
                DiceLoss(to_onehot_y=True, smooth_nr=1.0, smooth_dr=1.0)(channels_first(probs), labels_first(labels))
            ),
            smooth=1.0,
        ),
        case(
            "dice/class-weights",
            "dice",
            [probs, labels, weights],
            to_numpy(DiceLoss(to_onehot_y=True, weight=to_torch(weights))(channels_first(probs), labels_first(labels))),
        ),
        case(
            "tversky/dice-equivalent",
            "tversky",
            [probs, labels],
            to_numpy(TverskyLoss(to_onehot_y=True)(channels_first(probs), labels_first(labels))),
        ),
    ]
    ti = TverskyLoss(to_onehot_y=True, alpha=0.7, beta=0.3, reduction="none")(
        channels_first(probs), labels_first(labels)
    )
    out.append(
        case(
            "focalTversky",
            "focalTversky",
            [probs, labels],
            np.asarray((monai_none(ti) ** 0.75).mean()),
        )
    )
    out.append(
        case(
            "focalTversky/gamma",
            "focalTversky",
            [probs, labels],
            np.asarray((monai_none(ti) ** 2).sum()),
            gamma=2.0,
            reduction="sum",
        )
    )
    binary_logits = g.normal(size=(3, 4, 4)) * 2
    binary = (g.uniform(size=(3, 4, 4)) > 0.5).astype(np.float64)
    out += [
        case(
            "lovaszHinge/per-image",
            "lovaszHinge",
            [binary_logits, binary],
            np.asarray(
                np.mean(
                    [lovasz_hinge_flat(lg.ravel(), lb.ravel()) for lg, lb in zip(binary_logits, binary, strict=True)]
                )
            ),
        ),
        case(
            "lovaszHinge/pooled",
            "lovaszHinge",
            [binary_logits, binary],
            np.asarray(lovasz_hinge_flat(binary_logits.ravel(), binary.ravel())),
            perImage=False,
        ),
        case(
            "lovaszSoftmax",
            "lovaszSoftmax",
            [probs, labels],
            np.asarray(lovasz_softmax_flat(probs.reshape(-1, c), labels.ravel(), "present")),
        ),
        case(
            "lovaszSoftmax/per-image-all",
            "lovaszSoftmax",
            [probs, labels],
            np.asarray(
                np.mean(
                    [
                        lovasz_softmax_flat(p.reshape(-1, c), lb.ravel(), "all")
                        for p, lb in zip(probs, labels, strict=True)
                    ]
                )
            ),
            perImage=True,
            classes="all",
        ),
    ]
    return out
