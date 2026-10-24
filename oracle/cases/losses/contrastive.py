"""Contrastive and metric-learning losses against torch.nn.functional and optax (float64).

InfoNCE, supervised contrastive, CLIP, ArcFace and CosFace are written with torch operations exactly as in their
reference implementations (the papers' code); the Hadsell contrastive loss is its formula in numpy.
"""

import math

import jax
import numpy as np
import optax
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {"torch": torch.__version__, "optax": optax.__version__}


def info_nce(q: FloatArray, k: FloatArray, negatives: FloatArray | None, temperature: float) -> FloatArray:
    tq, tk = F.normalize(to_torch(q), dim=-1), F.normalize(to_torch(k), dim=-1)
    if negatives is None:
        logits = tq @ tk.T / temperature
        labels = torch.arange(len(q))
    else:
        tn = F.normalize(to_torch(negatives), dim=-1)
        pos = (tq * tk).sum(-1, keepdim=True)
        neg = tq @ tn.T if tn.ndim == 2 else (tq.unsqueeze(1) * tn).sum(-1)
        logits = torch.cat([pos, neg], dim=1) / temperature
        labels = torch.zeros(len(q), dtype=torch.long)
    return to_numpy(F.cross_entropy(logits, labels))


def supcon(features: FloatArray, labels: FloatArray, temperature: float, base_temperature: float) -> FloatArray:
    """SupConLoss (Khosla et al., 2020, the authors' code), one view per sample, anchors without positives dropped."""
    f = F.normalize(to_torch(features), dim=-1)
    lab = torch.tensor(labels).view(-1, 1)
    mask = torch.eq(lab, lab.T).double()
    logits = f @ f.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1 - torch.eye(len(f), dtype=torch.float64)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    counts = mask.sum(1)
    keep = counts > 0
    mean_log_prob = (mask * log_prob).sum(1)[keep] / counts[keep]
    return to_numpy((-(temperature / base_temperature) * mean_log_prob).mean())


def clip(img: FloatArray, txt: FloatArray, scale: float) -> FloatArray:
    i, t = F.normalize(to_torch(img), dim=-1), F.normalize(to_torch(txt), dim=-1)
    logits = scale * i @ t.T
    labels = torch.arange(len(img))
    return to_numpy((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2)


def arcface_logits(x: FloatArray, w: FloatArray, labels: FloatArray, s: float, m: float, easy: bool) -> torch.Tensor:
    """ArcMarginProduct of the insightface PyTorch reference."""
    cosine = F.normalize(to_torch(x)) @ F.normalize(to_torch(w)).T
    sine = torch.sqrt((1.0 - cosine**2).clamp(0, 1))
    phi = cosine * math.cos(m) - sine * math.sin(m)
    if easy:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
    one_hot = F.one_hot(torch.tensor(labels, dtype=torch.long), w.shape[0]).double()
    return s * (one_hot * phi + (1 - one_hot) * cosine)


def cosface_logits(x: FloatArray, w: FloatArray, labels: FloatArray, s: float, m: float) -> torch.Tensor:
    cosine = F.normalize(to_torch(x)) @ F.normalize(to_torch(w)).T
    one_hot = F.one_hot(torch.tensor(labels, dtype=torch.long), w.shape[0]).double()
    return s * (cosine - one_hot * m)


def cases() -> list[Case]:
    g = rng(45)
    n, d = 6, 4
    a, p, q = g.normal(size=(n, d)), g.normal(size=(n, d)), g.normal(size=(n, d))
    shared_neg, per_neg = g.normal(size=(5, d)), g.normal(size=(n, 3, d))
    labels = np.array([0.0, 1, 0, 2, 1, 0])
    views = np.concatenate([a, a + 0.3 * g.normal(size=(n, d))])
    view_labels = np.concatenate([np.arange(n), np.arange(n)]).astype(np.float64)
    same = (g.uniform(size=n) > 0.5).astype(np.float64)
    pm = np.where(same > 0, 1.0, -1.0)
    dist = np.abs(g.normal(size=n)) * 1.5
    centres = g.normal(size=(3, d))
    dd = np.sqrt(((a - p) ** 2).sum(-1))
    hadsell = 0.5 * (same * dd**2 + (1 - same) * np.maximum(0, 1.5 - dd) ** 2)
    out: list[Case] = [
        case("infoNce/in-batch", "infoNce", [a, p], info_nce(a, p, None, 0.1)),
        case("infoNce/temperature", "infoNce", [a, p], info_nce(a, p, None, 0.5), temperature=0.5),
        case("infoNce/shared", "infoNce", [a, p, shared_neg], info_nce(a, p, shared_neg, 0.2), temperature=0.2),
        case("infoNce/per-query", "infoNce", [a, p, per_neg], info_nce(a, p, per_neg, 0.2), temperature=0.2),
        case(
            "ntXent/labels",
            "ntXent",
            [a, labels],
            np.asarray(optax.losses.ntxent(a, labels.astype(np.int64), temperature=0.3), dtype=np.float64),
            temperature=0.3,
        ),
        case(
            "ntXent/views",
            "ntXent",
            [views, view_labels],
            np.asarray(optax.losses.ntxent(views, view_labels.astype(np.int64)), dtype=np.float64),
        ),
        case(
            "tripletMargin",
            "tripletMargin",
            [a, p, q],
            to_numpy(F.triplet_margin_loss(to_torch(a), to_torch(p), to_torch(q), margin=0.5)),
            margin=0.5,
        ),
        case(
            "tripletMargin/p1-swap",
            "tripletMargin",
            [a, p, q],
            to_numpy(F.triplet_margin_loss(to_torch(a), to_torch(p), to_torch(q), p=1, swap=True, reduction="none")),
            p=1,
            swap=True,
            reduction="none",
        ),
        case("contrastive", "contrastive", [a, p, same], hadsell, margin=1.5, reduction="none"),
        case(
            "cosineEmbedding",
            "cosineEmbedding",
            [a, p, pm],
            to_numpy(F.cosine_embedding_loss(to_torch(a), to_torch(p), to_torch(pm), margin=0.2)),
            margin=0.2,
        ),
        case(
            "hingeEmbedding",
            "hingeEmbedding",
            [dist, pm],
            to_numpy(F.hinge_embedding_loss(to_torch(dist), to_torch(pm), margin=1.2, reduction="sum")),
            margin=1.2,
            reduction="sum",
        ),
        case("supervisedContrastive", "supervisedContrastive", [a, labels], supcon(a, labels, 0.07, 0.07)),
        case(
            "supervisedContrastive/temperature",
            "supervisedContrastive",
            [a, labels],
            supcon(a, labels, 0.5, 0.07),
            temperature=0.5,
        ),
        case("clipLoss", "clipLoss", [a, p], clip(a, p, 1 / 0.07)),
        case("clipLoss/scale", "clipLoss", [a, p], clip(a, p, 10.0), logitScale=10.0),
        case(
            "arcFaceLogits",
            "arcFaceLogits",
            [a, centres, np.array([0.0, 1, 2, 0, 1, 2])],
            to_numpy(arcface_logits(a, centres, np.array([0.0, 1, 2, 0, 1, 2]), 30.0, 0.5, False)),
            scale=30.0,
            margin=0.5,
        ),
        case(
            "arcFaceLogits/easy",
            "arcFaceLogits",
            [a, centres, np.array([2.0, 1, 0, 0, 1, 2])],
            to_numpy(arcface_logits(a, centres, np.array([2.0, 1, 0, 0, 1, 2]), 64.0, 0.3, True)),
            margin=0.3,
            easyMargin=True,
        ),
        case(
            "arcFace",
            "arcFace",
            [a, centres, np.array([0.0, 1, 2, 0, 1, 2])],
            to_numpy(
                F.cross_entropy(
                    arcface_logits(a, centres, np.array([0.0, 1, 2, 0, 1, 2]), 16.0, 0.4, False),
                    torch.tensor([0, 1, 2, 0, 1, 2]),
                )
            ),
            scale=16.0,
            margin=0.4,
        ),
        case(
            "cosFace",
            "cosFace",
            [a, centres, np.array([0.0, 1, 2, 0, 1, 2])],
            to_numpy(
                F.cross_entropy(
                    cosface_logits(a, centres, np.array([0.0, 1, 2, 0, 1, 2]), 30.0, 0.35),
                    torch.tensor([0, 1, 2, 0, 1, 2]),
                )
            ),
            scale=30.0,
        ),
    ]
    return out
