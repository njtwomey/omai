"""Heteroscedastic empirical Bayes normal means against statsmodels' DerSimonian-Laird random-effects meta-analysis."""

import numpy as np
import statsmodels
from omai_oracle.fixture import Case, case
from statsmodels.stats.meta_analysis import combine_effects

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, p in [(1, 8), (2, 15)]:
        rng = np.random.default_rng(seed)
        variance = rng.uniform(0.2, 2.5, size=p)
        theta = rng.normal(0.5, 2.5, size=p)
        y = theta + rng.normal(size=p) * np.sqrt(variance)
        res = combine_effects(y, variance, method_re="dl")
        tau2 = float(res.tau2)  # pyright: ignore[reportAttributeAccessIssue]
        assert tau2 > 0  # DerSimonian-Laird is truncated at zero by the library; keep the cases away from the boundary
        mean = float(res.mean_effect_re)  # pyright: ignore[reportAttributeAccessIssue]
        shrink = variance / (variance + tau2)
        posterior = (1 - shrink) * y + shrink * mean
        out.append(
            case(
                f"dl/{seed}",
                "empiricalBayes",
                [y, variance],
                np.concatenate([[tau2, mean], posterior]),
                rtol=1e-8,
                atol=1e-10,
            )
        )
    return out
