"""Noise-robust classification losses against their reference formulas written with torch operations (float64).

Generalised cross-entropy follows the authors' code (Zhang and Sabuncu, 2018), symmetric cross-entropy the authors'
(Wang et al., 2019: predictions clamped at 1e-7, labels at 1e-4), the forward and backward corrections the formulas of
Patrini et al. (2017), and the bi-tempered logistic loss is a port of Google's reference implementation (the
fixed-point tempered softmax with five iterations).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def gce(logits: FloatArray, labels: FloatArray, q: float) -> FloatArray:
    p = F.softmax(to_torch(logits), dim=-1)
    py = p.gather(-1, torch.tensor(labels, dtype=torch.long).unsqueeze(-1)).squeeze(-1)
    return to_numpy(((1 - py**q) / q).mean())


def truncated_gce(logits: FloatArray, labels: FloatArray, q: float, k: float) -> FloatArray:
    p = F.softmax(to_torch(logits), dim=-1)
    py = p.gather(-1, torch.tensor(labels, dtype=torch.long).unsqueeze(-1)).squeeze(-1)
    loss = torch.where(py > k, (1 - py**q) / q, torch.full_like(py, (1 - k**q) / q))
    return to_numpy(loss.mean())


def sce(logits: FloatArray, labels: FloatArray, alpha: float, beta: float) -> FloatArray:
    """The authors' SCELoss: alpha CE + beta RCE with the prediction clamped at 1e-7 and the label at 1e-4."""
    z = to_torch(logits)
    y = torch.tensor(labels, dtype=torch.long)
    ce = F.cross_entropy(z, y)
    pred = F.softmax(z, dim=1).clamp(min=1e-7, max=1.0)
    hot = F.one_hot(y, z.shape[1]).double().clamp(min=1e-4, max=1.0)
    rce = -(pred * torch.log(hot)).sum(dim=1)
    return to_numpy(alpha * ce + beta * rce.mean())


def bootstrap(logits: FloatArray, labels: FloatArray, beta: float, hard: bool) -> FloatArray:
    z = to_torch(logits)
    logp = F.log_softmax(z, dim=-1)
    hot = F.one_hot(torch.tensor(labels, dtype=torch.long), z.shape[1]).double()
    belief = F.one_hot(logp.argmax(-1), z.shape[1]).double() if hard else logp.exp().detach()
    return to_numpy(-((beta * hot + (1 - beta) * belief) * logp).sum(-1).mean())


def forward(logits: FloatArray, labels: FloatArray, t: FloatArray) -> FloatArray:
    p = F.softmax(to_torch(logits), dim=-1) @ to_torch(t)
    return to_numpy(-torch.log(p.gather(-1, torch.tensor(labels, dtype=torch.long).unsqueeze(-1))).mean())


def backward(logits: FloatArray, labels: FloatArray, t: FloatArray) -> FloatArray:
    losses = -F.log_softmax(to_torch(logits), dim=-1)
    corrected = losses @ torch.linalg.inv(to_torch(t)).T
    return to_numpy(corrected.gather(-1, torch.tensor(labels, dtype=torch.long).unsqueeze(-1)).mean())


def log_t(u: FloatArray, t: float) -> FloatArray:
    return np.log(u) if t == 1.0 else (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u: FloatArray, t: float) -> FloatArray:
    return np.exp(u) if t == 1.0 else np.maximum(1.0 + (1.0 - t) * u, 0.0) ** (1.0 / (1.0 - t))


def tempered_softmax(a: FloatArray, t: float, iters: int) -> FloatArray:
    """Google's compute_normalization_fixed_point."""
    if t == 1.0:
        e = np.exp(a - a.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    mu = a.max(-1, keepdims=True)
    step0 = a - mu
    norm = step0
    for _ in range(iters):
        part = exp_t(norm, t).sum(-1, keepdims=True)
        norm = step0 * part ** (1.0 - t)
    part = exp_t(norm, t).sum(-1, keepdims=True)
    return exp_t(a - (-log_t(1.0 / part, t) + mu), t)


def bitempered(a: FloatArray, labels: FloatArray, t1: float, t2: float, iters: int) -> FloatArray:
    """Google's bi_tempered_logistic_loss for one-hot labels."""
    p = tempered_softmax(a, t2, iters)
    y = np.eye(a.shape[1])[labels.astype(int)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(y > 0, y * log_t(np.where(y > 0, y, 1.0), t1), 0.0)
    loss = ylog - y * log_t(p, t1) - (y ** (2.0 - t1) - p ** (2.0 - t1)) / (2.0 - t1)
    return np.asarray(loss.sum(-1).mean(), dtype=np.float64)


def cases() -> list[Case]:
    g = rng(51)
    z = g.normal(size=(7, 4)) * 1.5
    y = np.array([0.0, 3, 1, 2, 3, 0, 1])
    t = np.array([[0.7, 0.1, 0.1, 0.1], [0.05, 0.8, 0.1, 0.05], [0.1, 0.1, 0.7, 0.1], [0.2, 0.0, 0.1, 0.7]])
    out: list[Case] = [
        case("gce/q0.7", "generalisedCrossEntropy", [z, y], gce(z, y, 0.7), q=0.7),
        case("gce/q0.3", "generalisedCrossEntropy", [z, y], gce(z, y, 0.3), q=0.3),
        case("truncated", "truncatedCrossEntropy", [z, y], truncated_gce(z, y, 0.7, 0.3), q=0.7, k=0.3),
        case("symmetric", "symmetricCrossEntropy", [z, y], sce(z, y, 0.1, 1.0), alpha=0.1, beta=1.0, A=math.log(1e-4)),
        case(
            "symmetric/weights",
            "symmetricCrossEntropy",
            [z, y],
            sce(z, y, 1.0, 0.5),
            alpha=1.0,
            beta=0.5,
            A=math.log(1e-4),
        ),
        case("bootstrapped/soft", "bootstrapped", [z, y], bootstrap(z, y, 0.8, False), beta=0.8, mode="soft"),
        case("bootstrapped/hard", "bootstrapped", [z, y], bootstrap(z, y, 0.6, True), beta=0.6, mode="hard"),
        case("forwardCorrected", "forwardCorrected", [z, y, t], forward(z, y, t)),
        case("backwardCorrected", "backwardCorrected", [z, y, t], backward(z, y, t), rtol=1e-10),
    ]
    for t1, t2 in [(1.0, 1.0), (0.8, 1.2), (0.5, 1.5), (0.9, 2.0)]:
        out.append(
            case(
                f"biTempered/{t1}-{t2}",
                "biTemperedLogistic",
                [z, y],
                bitempered(z, y, t1, t2, 5),
                rtol=1e-10,
                t1=t1,
                t2=t2,
                iterations=5,
            )
        )
    return out
