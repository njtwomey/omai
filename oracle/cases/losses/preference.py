"""Preference objectives against their reference formulas written with torch operations (float64).

DPO, IPO and KTO follow the formulas of Hugging Face TRL's trainers; ORPO the authors' odds-ratio term; SimPO and the
Bradley-Terry reward-model loss are their formulas.
"""

import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def dpo(pc: FloatArray, pr: FloatArray, rc: FloatArray, rr: FloatArray, beta: float, smoothing: float) -> FloatArray:
    logits = (to_torch(pc) - to_torch(pr)) - (to_torch(rc) - to_torch(rr))
    losses = -F.logsigmoid(beta * logits) * (1 - smoothing) - F.logsigmoid(-beta * logits) * smoothing
    return to_numpy(losses.mean())


def ipo(pc: FloatArray, pr: FloatArray, rc: FloatArray, rr: FloatArray, tau: float) -> FloatArray:
    logits = (to_torch(pc) - to_torch(pr)) - (to_torch(rc) - to_torch(rr))
    return to_numpy(((logits - 1 / (2 * tau)) ** 2).mean())


def kto(
    pc: FloatArray, pr: FloatArray, rc: FloatArray, rr: FloatArray, beta: float, lam_u: float, kl: float
) -> FloatArray:
    chosen = to_torch(pc) - to_torch(rc)
    rejected = to_torch(pr) - to_torch(rr)
    good = 1 - torch.sigmoid(beta * (chosen - kl))
    bad = lam_u * (1 - torch.sigmoid(beta * (kl - rejected)))
    return to_numpy(torch.cat([good, bad]).mean())


def orpo(c: FloatArray, r: FloatArray, lam: float) -> FloatArray:
    """The authors' loss: NLL of the chosen plus lambda times -logsigmoid of the log odds ratio."""
    tc, tr = to_torch(c), to_torch(r)
    log_odds = (tc - tr) - (torch.log1p(-torch.exp(tc)) - torch.log1p(-torch.exp(tr)))
    return to_numpy((-tc - lam * F.logsigmoid(log_odds)).mean())


def simpo(c: FloatArray, r: FloatArray, beta: float, gamma: float) -> FloatArray:
    return to_numpy((-F.logsigmoid(beta * (to_torch(c) - to_torch(r)) - gamma)).mean())


def bradley_terry(c: FloatArray, r: FloatArray, margin: float) -> FloatArray:
    return to_numpy((-F.logsigmoid(to_torch(c) - to_torch(r) - margin)).mean())


def cases() -> list[Case]:
    g = rng(54)
    n = 9
    pc, pr = -g.uniform(2, 12, size=n), -g.uniform(2, 12, size=n)
    rc, rr = -g.uniform(2, 12, size=n), -g.uniform(2, 12, size=n)
    avg_c, avg_r = -g.uniform(0.05, 2.5, size=n), -g.uniform(0.05, 2.5, size=n)
    out: list[Case] = [
        case("dpo", "dpo", [pc, pr, rc, rr], dpo(pc, pr, rc, rr, 0.1, 0.0), beta=0.1),
        case("dpo/smoothing", "dpo", [pc, pr, rc, rr], dpo(pc, pr, rc, rr, 0.3, 0.15), beta=0.3, labelSmoothing=0.15),
        case("ipo", "ipo", [pc, pr, rc, rr], ipo(pc, pr, rc, rr, 0.2), tau=0.2),
        case(
            "kto",
            "kto",
            [pc, pr, rc, rr],
            kto(pc, pr, rc, rr, 0.2, 1.5, 0.3),
            beta=0.2,
            lambdaU=1.5,
            referencePoint=0.3,
        ),
        case("orpo", "orpo", [avg_c, avg_r], orpo(avg_c, avg_r, 0.25), lambda_=0.25),
        case("simpo", "simpo", [avg_c, avg_r], simpo(avg_c, avg_r, 2.0, 0.5), beta=2.0, gamma=0.5),
        case("bradleyTerry", "bradleyTerry", [pc, pr], bradley_terry(pc, pr, 0.0)),
        case("bradleyTerry/margin", "bradleyTerry", [pc, pr], bradley_terry(pc, pr, 0.4), margin=0.4),
    ]
    return out
