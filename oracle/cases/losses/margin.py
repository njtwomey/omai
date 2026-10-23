"""Margin losses against optax, torch.nn.functional and sklearn.metrics (float64)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sklearn
import sklearn.metrics as M
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {
    "optax": optax.__version__,
    "torch": torch.__version__,
    "scikit-learn": sklearn.__version__,
}


def from_jax(x: jax.Array) -> FloatArray:
    """A float64 numpy copy of a jax array."""
    return np.asarray(x, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(42)
    f = g.normal(size=(4, 5)) * 1.5
    y = np.where(g.uniform(size=(4, 5)) > 0.5, 1.0, -1.0)
    scores = g.normal(size=(6, 4)) * 2
    labels = np.array([0.0, 3, 1, 2, 3, 1])
    w = np.array([0.5, 1.0, 2.0, 1.5])
    z = y * f
    out: list[Case] = [
        case("hinge", "hinge", [f, y], from_jax(jnp.mean(optax.losses.hinge_loss(f, y)))),
        case("hinge/none", "hinge", [f, y], from_jax(optax.losses.hinge_loss(f, y)), reduction="none"),
        case("squaredHinge", "squaredHinge", [f, y], np.asarray((np.maximum(0, 1 - z) ** 2).mean())),
        case(
            "squaredHinge/margin",
            "squaredHinge",
            [f, y],
            np.asarray((np.maximum(0, 2 - z) ** 2).sum()),
            margin=2,
            reduction="sum",
        ),
        case("logistic", "logistic", [f, y], to_numpy(F.soft_margin_loss(to_torch(f), to_torch(y)))),
        case(
            "logistic/sum",
            "logistic",
            [f, y],
            to_numpy(F.soft_margin_loss(to_torch(f), to_torch(y), reduction="sum")),
            reduction="sum",
        ),
        case("exponential", "exponential", [f, y], np.asarray(np.exp(-z).mean())),
        case(
            "modifiedHuber",
            "modifiedHuber",
            [f, y],
            np.where(z >= -1, np.maximum(0, 1 - z) ** 2, -4 * z),
            reduction="none",
        ),
        case("perceptron", "perceptron", [f, y], from_jax(jnp.mean(optax.losses.perceptron_loss(f, y)))),
        case(
            "crammerSinger",
            "crammerSinger",
            [scores, labels],
            from_jax(optax.losses.multiclass_hinge_loss(scores, labels.astype(np.int64))),
            reduction="none",
        ),
        case(
            "crammerSinger/sklearn",
            "crammerSinger",
            [scores, labels],
            np.asarray(M.hinge_loss(labels.astype(np.int64), scores, labels=[0, 1, 2, 3])),
        ),
    ]
    long_labels = torch.tensor(labels, dtype=torch.long)
    for p in [1, 2]:
        for reduction in ["mean", "sum", "none"]:
            out.append(
                case(
                    f"westonWatkins/p={p}/{reduction}",
                    "westonWatkins",
                    [scores, labels],
                    to_numpy(F.multi_margin_loss(to_torch(scores), long_labels, p=p, margin=0.8, reduction=reduction)),
                    p=p,
                    margin=0.8,
                    reduction=reduction,
                )
            )
    out.append(
        case(
            "westonWatkins/weighted",
            "westonWatkins",
            [scores, labels, w],
            to_numpy(F.multi_margin_loss(to_torch(scores), long_labels, weight=to_torch(w))),
        )
    )
    return out
