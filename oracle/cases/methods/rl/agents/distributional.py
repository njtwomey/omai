"""Distributional RL kernels against numpy transcriptions of the papers: the C51 categorical projection (Bellemare,
Dabney and Munos 2017, Algorithm 1) and the quantile Huber loss of QR-DQN and IQN (Dabney et al. 2018, 2018b)."""

import numpy as np
import numpy.typing as npt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

type Array = npt.NDArray[np.float64]


def projection(probs: Array, rewards: Array, discounts: Array, v_min: float, v_max: float) -> Array:
    """Algorithm 1 of C51: move each atom to r + d z_j, clip, and split its mass between the neighbouring atoms."""
    n = probs.shape[1]
    z = np.linspace(v_min, v_max, n)
    dz = (v_max - v_min) / (n - 1)
    out = np.zeros_like(probs)
    for b in range(probs.shape[0]):
        tz = np.clip(rewards[b] + discounts[b] * z, v_min, v_max)
        pos = np.clip((tz - v_min) / dz, 0, n - 1)
        lower = np.floor(pos).astype(np.int64)
        upper = np.ceil(pos).astype(np.int64)
        for j in range(n):
            if lower[j] == upper[j]:
                out[b, lower[j]] += probs[b, j]
            else:
                out[b, lower[j]] += probs[b, j] * (upper[j] - pos[j])
                out[b, upper[j]] += probs[b, j] * (pos[j] - lower[j])
    return out


def quantile_huber(theta: Array, targets: Array, taus: Array, kappa: float) -> Array:
    """Per-sample sum over predicted quantiles of the mean over targets of |tau - 1{u < 0}| L_kappa(u) / kappa."""
    u = targets[:, None, :] - theta[:, :, None]
    a = np.abs(u)
    huber = np.where(a <= kappa, 0.5 * u**2, kappa * (a - 0.5 * kappa)) / kappa if kappa > 0 else a
    weight = np.abs(taus[:, :, None] - (u < 0).astype(np.float64))
    return (weight * huber).mean(axis=2).sum(axis=1)


def softmax_rows(x: Array) -> Array:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cases() -> list[Case]:
    g = rng(51)
    out: list[Case] = []
    for i, (atoms, v_min, v_max) in enumerate([(5, -1.0, 1.0), (11, -10.0, 10.0), (51, -10.0, 10.0), (21, 0.0, 3.0)]):
        batch = 6
        probs = softmax_rows(g.normal(size=(batch, atoms)) * 2)
        rewards = g.normal(size=batch) * (v_max - v_min) / 4
        discounts = np.array([0.99, 0.9, 0.0, 0.5, 0.99**3, 1.0])
        out.append(
            case(
                f"projection/{i}",
                "categoricalProjection",
                [probs, rewards, discounts],
                projection(probs, rewards, discounts, v_min, v_max),
                vMin=v_min,
                vMax=v_max,
            )
        )
    shapes = [(4, 4, 1.0, False), (8, 5, 1.0, True), (6, 6, 0.0, False), (5, 7, 2.5, True)]
    for i, (n, m, kappa, per_sample_taus) in enumerate(shapes):
        batch = 5
        theta = g.normal(size=(batch, n)) * 2
        targets = g.normal(size=(batch, m)) * 2
        taus = g.uniform(size=(batch, n)) if per_sample_taus else np.tile((2 * np.arange(n) + 1) / (2 * n), (batch, 1))
        out.append(
            case(
                f"quantileHuber/{i}",
                "quantileHuberLoss",
                [theta, targets, taus],
                quantile_huber(theta, targets, taus, kappa),
                kappa=kappa,
            )
        )
    return out
