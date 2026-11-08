"""Noise-contrastive estimation and vector quantisation against their reference formulas in torch (float64).

The VQ-VAE loss follows the widely used PyTorch implementation (codebook MSE against the detached latents plus beta
times commitment MSE against the detached codes); the EMA codebook follows Sonnet's VectorQuantizerEMA.
"""

import numpy as np
import torch
import torch.nn.functional as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def nce(sd: FloatArray, qd: FloatArray, sn: FloatArray, qn: FloatArray, c: float) -> FloatArray:
    """Gutmann and Hyvarinen: -mean log sigmoid(G(data)) - (1 / N) sum log sigmoid(-G(noise)), k = M / N."""
    k = len(sn) / len(sd)
    gd = to_torch(sd) - c - np.log(k) - to_torch(qd)
    gn = to_torch(sn) - c - np.log(k) - to_torch(qn)
    return to_numpy(-F.logsigmoid(gd).mean() - F.logsigmoid(-gn).sum() / len(sd))


def vq_loss(z: FloatArray, book: FloatArray, beta: float) -> tuple[FloatArray, FloatArray]:
    zt, cb = to_torch(z), to_torch(book)
    idx = torch.cdist(zt, cb).argmin(dim=1)
    e = cb[idx]
    loss = F.mse_loss(e, zt.detach()) + beta * F.mse_loss(zt, e.detach())
    return to_numpy(loss), idx.numpy().astype(np.float64)


def ema_update(
    counts: FloatArray, sums: FloatArray, z: FloatArray, idx: FloatArray, decay: float, eps: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    k = len(counts)
    onehot = F.one_hot(torch.tensor(idx, dtype=torch.long), k).double()
    new_counts = to_torch(counts) * decay + (1 - decay) * onehot.sum(0)
    total = new_counts.sum()
    smoothed = (new_counts + eps) / (total + k * eps) * total
    new_sums = to_torch(sums) * decay + (1 - decay) * onehot.T @ to_torch(z)
    return to_numpy(new_counts), to_numpy(new_sums), to_numpy(new_sums / smoothed.unsqueeze(1))


def cases() -> list[Case]:
    g = rng(56)
    sd, sn = g.normal(size=7), g.normal(size=14)
    qd, qn = g.normal(size=7) - 1.0, g.normal(size=14) - 1.0
    z, book = g.normal(size=(10, 3)), g.normal(size=(6, 3))
    loss, idx = vq_loss(z, book, 0.25)
    counts, sums = np.abs(g.normal(size=6)) + 0.5, g.normal(size=(6, 3))
    new_counts, new_sums, new_book = ema_update(counts, sums, z, idx, 0.99, 1e-5)
    return [
        case("nce", "noiseContrastive", [sd, qd, sn, qn], nce(sd, qd, sn, qn, 0.0)),
        case("nce/normaliser", "noiseContrastive", [sd, qd, sn, qn], nce(sd, qd, sn, qn, 0.7), logNormaliser=0.7),
        case("vq/loss", "vectorQuantiseLoss", [z, book], loss, beta=0.25),
        case("vq/indices", "vectorQuantiseIndices", [z, book], idx),
        case("vq/ema-counts", "vqEmaCounts", [counts, sums, z, idx], new_counts, decay=0.99, eps=1e-5),
        case("vq/ema-sums", "vqEmaSums", [counts, sums, z, idx], new_sums, decay=0.99, eps=1e-5),
        case("vq/ema-codebook", "vqEmaCodebook", [counts, sums, z, idx], new_book, decay=0.99, eps=1e-5),
    ]
