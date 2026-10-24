"""Generative-model losses against torch: GAN objectives and gradient penalties (torch.autograd), ELBO terms
(torch.distributions), denoising score matching, diffusion and flow-matching objectives written with torch operations.
"""

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}

OBJECTIVES = ["minimax", "non-saturating", "wasserstein", "hinge", "least-squares"]


def d_loss(r: torch.Tensor, f: torch.Tensor, objective: str) -> torch.Tensor:
    if objective in ("minimax", "non-saturating"):
        return F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + F.binary_cross_entropy_with_logits(
            f, torch.zeros_like(f)
        )
    if objective == "wasserstein":
        return f.mean() - r.mean()
    if objective == "hinge":
        return F.relu(1 - r).mean() + F.relu(1 + f).mean()
    return 0.5 * ((r - 1) ** 2).mean() + 0.5 * (f**2).mean()


def g_loss(f: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "minimax":
        return -F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    if objective == "non-saturating":
        return F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    if objective in ("wasserstein", "hinge"):
        return -f.mean()
    return 0.5 * ((f - 1) ** 2).mean()


def critic_grad_sq(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Squared input-gradient norms of the critic D(x) = tanh(x W) v, per sample."""
    x = x.clone().requires_grad_(True)
    scores = torch.tanh(x @ w) @ v
    (g,) = torch.autograd.grad(scores.sum(), x)
    return (g**2).sum(-1)


def cases() -> list[Case]:
    g = rng(47)
    real, fake = g.normal(size=6), g.normal(size=6)
    tr, tf = to_torch(real), to_torch(fake)
    out: list[Case] = []
    for objective in OBJECTIVES:
        out.append(
            case(
                f"discriminator/{objective}",
                "discriminatorLoss",
                [real, fake],
                to_numpy(d_loss(tr, tf, objective)),
                objective=objective,
            )
        )
        out.append(
            case(
                f"generator/{objective}",
                "generatorLoss",
                [fake, fake],
                to_numpy(g_loss(tf, objective)),
                objective=objective,
            )
        )
    xr, xf = g.normal(size=(5, 3)), g.normal(size=(5, 3))
    w, v = g.normal(size=(3, 4)), g.normal(size=4)
    alpha = g.uniform(size=5)
    a = to_torch(alpha)[:, None]
    interp = a * to_torch(xr) + (1 - a) * to_torch(xf)
    gp = 10 * ((critic_grad_sq(interp, to_torch(w), to_torch(v)).sqrt() - 1) ** 2).mean()
    r1 = 5 * critic_grad_sq(to_torch(xr), to_torch(w), to_torch(v)).mean()
    out += [
        case("gradientPenalty", "gradientPenalty", [xr, xf, w, v, alpha], to_numpy(gp)),
        case("r1Penalty", "r1Penalty", [xr, w, v], to_numpy(r1)),
    ]
    mu, s = g.normal(size=(4, 3)), 0.5 * g.normal(size=(4, 3))
    q = D.Normal(to_torch(mu), torch.exp(to_torch(s) / 2))
    prior = D.Normal(torch.zeros(4, 3, dtype=torch.float64), torch.ones(4, 3, dtype=torch.float64))
    z = g.normal(size=(7, 4, 3))
    x01 = g.uniform(size=(4, 5))
    logits = g.normal(size=(4, 5))
    ll = -F.binary_cross_entropy_with_logits(to_torch(logits), to_torch(x01), reduction="none").sum(-1)
    xg, dec, dec_s = g.normal(size=(4, 5)), g.normal(size=(4, 5)), 0.3 * g.normal(size=(4, 5))
    gll = D.Normal(to_torch(dec), torch.exp(to_torch(dec_s) / 2)).log_prob(to_torch(xg)).sum(-1)
    logw = g.normal(size=(4, 6))
    kl = D.kl_divergence(q, prior).sum(-1)
    out += [
        case("gaussianKl", "gaussianKl", [mu, s], to_numpy(kl.mean())),
        case("gaussianKl/none", "gaussianKl", [mu, s], to_numpy(kl), reduction="none"),
        case(
            "sampledKl",
            "sampledKl",
            [mu, s, z],
            to_numpy((q.log_prob(to_torch(z)) - prior.log_prob(to_torch(z))).mean()),
        ),
        case("bernoulliLogLikelihood", "bernoulliLogLikelihood", [logits, x01], to_numpy(ll)),
        case("gaussianLogLikelihood", "gaussianLogLikelihood", [dec, xg, dec_s], to_numpy(gll)),
        case(
            "gaussianLogLikelihood/unit",
            "gaussianLogLikelihood",
            [dec, xg],
            to_numpy(D.Normal(to_torch(dec), 1.0).log_prob(to_torch(xg)).sum(-1)),
        ),
        case("negativeElbo", "negativeElbo", [to_numpy(ll), to_numpy(kl)], to_numpy((0.5 * kl - ll).mean()), beta=0.5),
        case(
            "importanceWeightedBound",
            "importanceWeightedBound",
            [logw, logw],
            to_numpy(-(torch.logsumexp(to_torch(logw), -1) - np.log(6)).mean()),
        ),
    ]
    score, eps = g.normal(size=(4, 2, 3)), g.normal(size=(4, 2, 3))
    sigma = np.exp(g.normal(size=4))
    ts, te, tsg = to_torch(score), to_torch(eps), to_torch(sigma)[:, None, None]
    dsm = (0.5 * ((ts + te / tsg) ** 2).sum((1, 2)) * tsg.squeeze() ** 2).mean()
    dsm_plain = (0.5 * ((ts + te / tsg) ** 2).sum((1, 2))).sum()
    x0, noise, pred = g.normal(size=(4, 3)), g.normal(size=(4, 3)), g.normal(size=(4, 3))
    al, sg = g.uniform(0.2, 0.9, size=(4, 1)), g.uniform(0.2, 0.9, size=(4, 1))
    tgt_v = al * noise - sg * x0
    t = g.uniform(size=4)
    xt = (1 - (1 - 0.01) * t[:, None]) * x0 + t[:, None] * pred
    out += [
        case("denoisingScoreMatching", "denoisingScoreMatching", [score, eps, sigma], to_numpy(dsm)),
        case(
            "denoisingScoreMatching/unweighted",
            "denoisingScoreMatching",
            [score, eps, sigma],
            to_numpy(dsm_plain),
            weighting="none",
            reduction="sum",
        ),
        case(
            "diffusionLoss/epsilon",
            "diffusionLoss",
            [pred, x0, noise, al, sg],
            np.asarray(((pred - noise) ** 2).mean()),
        ),
        case(
            "diffusionLoss/x0",
            "diffusionLoss",
            [pred, x0, noise, al, sg],
            np.asarray(((pred - x0) ** 2).mean()),
            parameterisation="x0",
        ),
        case(
            "diffusionLoss/v",
            "diffusionLoss",
            [pred, x0, noise, al, sg],
            np.asarray(((pred - tgt_v) ** 2).mean()),
            parameterisation="v",
        ),
        case("flowMatchingPath", "flowMatchingPath", [x0, pred, t], xt, sigmaMin=0.01),
        case(
            "flowMatching",
            "flowMatching",
            [noise, x0, pred],
            np.asarray(((noise - (pred - (1 - 0.01) * x0)) ** 2).mean()),
            sigmaMin=0.01,
        ),
    ]
    return out
