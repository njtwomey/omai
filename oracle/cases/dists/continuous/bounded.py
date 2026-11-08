"""Distributions on bounded intervals (dists/continuous/bounded.ts) against scipy.stats and torch.distributions."""

import numpy as np
import scipy
import torch
from omai_oracle.distributions import distribution_cases, method_case
from omai_oracle.fixture import Case
from omai_oracle.helpers import rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(35)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    unit = g.uniform(0.01, 0.99, 8)
    out: list[Case] = []
    lo, hi = np.array([-1.0]), np.array([3.0])
    out += distribution_cases(
        "uniform", "uniform", [lo, hi], stats.uniform(lo, hi - lo), g.uniform(-1, 3, 8), q, skip=("logCdf", "logSf")
    )
    for i, (a_, b_) in enumerate([(0.5, 0.5), (2.0, 5.0), (30.0, 3.0)]):
        a, b = np.array([a_]), np.array([b_])
        out += distribution_cases(f"beta/{i}", "beta", [a, b], stats.beta(a, b), unit, q, skip=("logCdf", "logSf"))
    for i, (lo_, c_, hi_) in enumerate([(0.0, 0.5, 1.0), (-2.0, -1.5, 4.0), (1.0, 3.0, 3.0)]):
        p = [np.array([lo_]), np.array([c_]), np.array([hi_])]
        frozen = stats.triang((p[1] - p[0]) / (p[2] - p[0]), loc=p[0], scale=p[2] - p[0])
        out += distribution_cases(
            f"triangular/{i}", "triangular", p, frozen, g.uniform(lo_, hi_, 8), q, skip=("logCdf", "logSf")
        )
    # Kumaraswamy against torch.distributions (scipy has none).
    for i, (a_, b_) in enumerate([(0.7, 1.5), (2.0, 5.0), (5.0, 0.8)]):
        a, b = np.array([a_]), np.array([b_])
        d = torch.distributions.Kumaraswamy(to_torch(a), to_torch(b))
        params = [a, b]
        label = f"kumaraswamy/{i}"
        log_prob = to_numpy(d.log_prob(to_torch(unit)))
        out += [
            method_case(label, "kumaraswamy", "logProb", params, unit, log_prob),
            method_case(label, "kumaraswamy", "mean", params, None, to_numpy(d.mean)),
            method_case(label, "kumaraswamy", "variance", params, None, to_numpy(d.variance)),
            method_case(label, "kumaraswamy", "entropy", params, None, to_numpy(d.entropy())),
        ]
    return out
