"""Invariance penalties against torch autograd and the reference code of Deep CORAL and Fishr (float64).

The IRMv1 penalty is the squared autograd derivative of the risk with respect to a dummy scale at one (the authors'
`penalty` function, with create_graph); Deep CORAL is the authors' covariance loss; the Fishr penalty uses per-sample
gradients by autograd and the population variance as BackPACK's `variance` extension.
"""

import torch
import torch.nn.functional as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def irm_penalty(logits: FloatArray, y: FloatArray, loss: str, unbiased: bool) -> FloatArray:
    """The authors' IRMv1 penalty (Arjovsky et al.), with the unbiased two-half variant of their code."""

    def risk(s: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if loss == "bce":
            return F.binary_cross_entropy_with_logits(s * w, t)
        return F.mse_loss(s * w, t)

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    s, t = to_torch(logits), to_torch(y)
    if not unbiased:
        (g,) = torch.autograd.grad(risk(s, t, scale), [scale], create_graph=True)
        return to_numpy(g**2)
    half = len(s) // 2
    (g1,) = torch.autograd.grad(risk(s[:half], t[:half], scale), [scale], create_graph=True)
    (g2,) = torch.autograd.grad(risk(s[half : 2 * half], t[half : 2 * half], scale), [scale], create_graph=True)
    return to_numpy(g1 * g2)


def deep_coral(source: FloatArray, target: FloatArray) -> FloatArray:
    d = source.shape[1]

    def cov(x: torch.Tensor) -> torch.Tensor:
        xm = x - x.mean(0, keepdim=True)
        return xm.T @ xm / (len(x) - 1)

    return to_numpy(((cov(to_torch(source)) - cov(to_torch(target))) ** 2).sum() / (4 * d * d))


def per_sample_logistic_gradients(x: FloatArray, y: FloatArray, w: FloatArray) -> FloatArray:
    """The gradient of the per-sample logistic loss in w for a linear model, by autograd, `[N, P]`."""
    rows = []
    for xi, yi in zip(to_torch(x), to_torch(y), strict=True):
        wt = to_torch(w).clone().requires_grad_(True)
        (g,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(xi @ wt, yi), [wt])
        rows.append(g)
    return to_numpy(torch.stack(rows))


def fishr(grads: list[FloatArray]) -> FloatArray:
    """Population variance of the per-sample gradients per environment, then the mean squared distance to their mean."""
    variances = [to_torch(g).pow(2).mean(0) - to_torch(g).mean(0).pow(2) for g in grads]
    mean_var = torch.stack(variances).mean(0)
    return to_numpy(torch.stack([(v - mean_var).pow(2).sum() for v in variances]).mean())


def cases() -> list[Case]:
    g = rng(55)
    n = 10
    logits = g.normal(size=n) * 2
    labels = (g.uniform(size=n) > 0.5).astype(float)
    reg_target = g.normal(size=n)
    source, target = g.normal(size=(12, 4)), g.normal(size=(9, 4)) * 1.7 + 0.3
    w = g.normal(size=3)
    envs = [(g.normal(size=(m, 3)), (g.uniform(size=m) > 0.5).astype(float)) for m in (8, 11, 6)]
    grads = [per_sample_logistic_gradients(x, y, w) for x, y in envs]
    return [
        case("irm/bce", "irmPenalty", [logits, labels], irm_penalty(logits, labels, "bce", False)),
        case("irm/mse", "irmPenalty", [logits, reg_target], irm_penalty(logits, reg_target, "mse", False), risk="mse"),
        case(
            "irm/unbiased",
            "irmPenalty",
            [logits, labels],
            irm_penalty(logits, labels, "bce", True),
            unbiased=True,
        ),
        case("coral", "coral", [source, target], deep_coral(source, target)),
        case("fishr", "fishrPenalty", grads, fishr(grads)),
    ]
