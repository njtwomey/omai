"""Relaxed distributions (dists/relaxed.ts) against torch.distributions."""

import numpy as np
import torch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(44)
    out: list[Case] = []
    for i, (t, logit) in enumerate([(0.7, -0.5), (2.0, 1.3)]):
        x = g.uniform(0.02, 0.98, 6)
        d = torch.distributions.RelaxedBernoulli(to_torch(np.array([t])), logits=to_torch(np.array([logit])))
        params = [np.array([t]), np.array([logit])]
        out.append(
            case(
                f"relaxedBernoulli/{i}/logProb",
                "relaxedBernoulli.logProb",
                [*params, x],
                to_numpy(d.log_prob(to_torch(x))),
                rtol=1e-12,
                arity=2,
            )
        )
    for i, t in enumerate([0.5, 1.5]):
        logits = g.normal(size=3)
        x = g.dirichlet(np.ones(3), size=5)
        d = torch.distributions.RelaxedOneHotCategorical(to_torch(np.array(t)), logits=to_torch(logits))
        out.append(
            case(
                f"relaxedOneHotCategorical/{i}/logProb",
                "relaxedOneHotCategorical.logProb",
                [np.array(t), logits, x],
                to_numpy(d.log_prob(to_torch(x))),
                rtol=1e-11,
                arity=2,
            )
        )
    return out
