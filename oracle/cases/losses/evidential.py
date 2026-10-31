"""Evidential losses against their reference formulas (torch special functions) and scipy's Student-t."""

import numpy as np
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from scipy import stats

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def kl_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)), Sensoy et al. (2018) eq. (6)."""
    k = alpha.shape[-1]
    s = alpha.sum(-1, keepdim=True)
    first = torch.lgamma(s.squeeze(-1)) - torch.lgamma(torch.tensor(float(k), dtype=torch.float64))
    second = torch.lgamma(alpha).sum(-1)
    third = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s))).sum(-1)
    return first - second + third


def evidential(evidence: FloatArray, labels: FloatArray, kind: str, kl_weight: float) -> FloatArray:
    e = to_torch(evidence)
    alpha = e + 1
    s = alpha.sum(-1, keepdim=True)
    y = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), e.shape[1]).double()
    if kind == "mse":
        p = alpha / s
        data = ((y - p) ** 2 + alpha * (s - alpha) / (s**2 * (s + 1))).sum(-1)
    elif kind == "log":
        data = (y * (torch.log(s) - torch.log(alpha))).sum(-1)
    else:
        data = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)
    tilde = y + (1 - y) * alpha
    return to_numpy((data + kl_weight * kl_uniform(tilde)).mean())


def cases() -> list[Case]:
    g = rng(52)
    e = np.abs(g.normal(size=(6, 4))) * 3
    y = np.array([0.0, 3, 1, 2, 3, 0])
    out: list[Case] = []
    for kind in ["mse", "log", "digamma"]:
        out.append(case(f"dirichlet/{kind}", "evidentialNll", [e, y], evidential(e, y, kind, 0.0), kind=kind))
        out.append(
            case(f"dirichlet/{kind}/kl", "evidentialNll", [e, y], evidential(e, y, kind, 0.3), kind=kind, klWeight=0.3)
        )
    # The normal-inverse-gamma NLL is the negative log density of a Student-t with 2 alpha degrees of freedom.
    n = 8
    gamma, y_obs = g.normal(size=n), g.normal(size=n)
    nu, alpha, beta = np.exp(g.normal(size=n) * 0.5), 1.1 + np.abs(g.normal(size=n)), np.exp(g.normal(size=n) * 0.5)
    scale = np.sqrt(beta * (1 + nu) / (nu * alpha))
    nll = -stats.t.logpdf(y_obs, df=2 * alpha, loc=gamma, scale=scale)
    out.append(case("nig/nll", "normalInverseGammaNll", [gamma, nu, alpha, beta, y_obs], nll, reduction="none"))
    reg = np.abs(y_obs - gamma) * (2 * nu + alpha)
    out.append(
        case(
            "nig/regularised",
            "normalInverseGammaNll",
            [gamma, nu, alpha, beta, y_obs],
            nll + 0.4 * reg,
            reduction="none",
            lambda_=0.4,
        )
    )
    return out
