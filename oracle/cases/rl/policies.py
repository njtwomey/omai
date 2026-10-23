"""Policy heads against torch.distributions (float64): categorical and diagonal Gaussian log-probabilities and
entropies, and tanh-squashed, rescaled Gaussians (TransformedDistribution with TanhTransform and AffineTransform)."""

import numpy as np
import scipy
import scipy.special
import torch
from torch.distributions import (
    AffineTransform,
    Categorical,
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(75)
    out: list[Case] = []
    logits = g.normal(size=(4, 5)) * 2
    actions = g.integers(0, 5, size=4).astype(np.float64)
    cat = Categorical(logits=to_torch(logits))
    out.append(
        case("categorical/logProb", "categoricalLogProb", [logits, actions], to_numpy(cat.log_prob(to_torch(actions))))
    )
    out.append(case("categorical/entropy", "categoricalEntropy", [logits], to_numpy(cat.entropy())))
    q = g.normal(size=6)
    out.append(case("boltzmann", "boltzmannProbabilities", [q], scipy.special.softmax(q / 0.7), temperature=0.7))
    mean = g.normal(size=(3, 2))
    log_std = g.normal(scale=0.5, size=(3, 2))
    a = g.normal(size=(3, 2))
    base = Independent(Normal(to_torch(mean), to_torch(np.exp(log_std))), 1)
    out.append(case("gaussian/logProb", "gaussianLogProb", [mean, log_std, a], to_numpy(base.log_prob(to_torch(a)))))
    out.append(case("gaussian/entropy", "gaussianEntropy", [mean, log_std], to_numpy(base.entropy())))
    squashed = TransformedDistribution(base, [TanhTransform()])
    u = np.tanh(g.normal(size=(3, 2)) * 0.8)
    out.append(
        case(
            "squashed/logProb",
            "squashedLogProb",
            [mean, log_std, u],
            to_numpy(squashed.log_prob(to_torch(u))),
            rtol=1e-9,
            atol=1e-9,
        )
    )
    low, high = np.array([-2.0, 0.0]), np.array([2.0, 3.0])
    rescaled = TransformedDistribution(
        base, [TanhTransform(), AffineTransform(to_torch((high + low) / 2), to_torch((high - low) / 2))]
    )
    act = (high + low) / 2 + (high - low) / 2 * u
    out.append(
        case(
            "rescaled/logProb",
            "rescaledLogProb",
            [mean, log_std, act],
            to_numpy(rescaled.log_prob(to_torch(act))),
            rtol=1e-9,
            atol=1e-9,
            low=low.tolist(),
            high=high.tolist(),
        )
    )
    return out
