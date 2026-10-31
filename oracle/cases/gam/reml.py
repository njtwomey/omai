"""REML smoothing selection against statsmodels' MixedLM: a random-effect term is a penalty of the identity with smoothing
parameter lambda = sigma^2 / sigma_b^2, so the REML optimum of the additive model is the REML fit of the random-intercept
mixed model. The engine's REML score differs from the mixed model's log-likelihood by constants only."""

import numpy as np
import statsmodels
import statsmodels.api as sm
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(616)
    out: list[Case] = []
    for name, sizes, sigma_b, sigma in [("balanced", [8] * 7, 1.2, 0.8), ("unbalanced", [3, 9, 5, 12, 4, 7, 10, 6], 0.7, 1.1)]:
        groups = np.concatenate([np.full(s, float(i)) for i, s in enumerate(sizes)])
        effects = sigma_b * g.normal(size=len(sizes))
        y = 2.0 + effects[groups.astype(int)] + sigma * g.normal(size=len(groups))
        model = sm.MixedLM(y, np.ones((len(y), 1)), groups=groups)
        fit = model.fit(reml=True, method="lbfgs", gtol=1e-10)
        sigma_b2 = float(np.asarray(fit.cov_re)[0, 0])
        sigma2 = float(fit.scale)
        out.append(
            case(
                name,
                "mixedModelReml",
                [groups, y],
                {
                    "lambda": np.asarray([sigma2 / sigma_b2]),
                    "sigma2": np.asarray([sigma2]),
                    "intercept": np.asarray([fit.fe_params[0]], dtype=np.float64),
                },
                rtol=2e-3,
                atol=1e-6,
            )
        )
    return out
