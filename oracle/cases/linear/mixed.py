"""Linear mixed models against statsmodels' MixedLM: fixed effects and their standard errors, the residual variance,
the random-effects covariance, the (restricted) log-likelihood and the group effects, for a random intercept and for a
random intercept with a correlated random slope, fitted by REML and by ML. statsmodels stops short of the optimum (its
log-likelihood is a few 1e-8 below the exact maximum), which moves the covariances by about 1e-4 relative and the
standard errors by up to 5e-4, hence the tolerance."""

import numpy as np
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.regression.mixed_linear_model import MixedLM

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(1982)
    out: list[Case] = []
    for name, groups, per, slope in [("intercept", 10, 8, False), ("slope", 14, 10, True)]:
        n = groups * per
        gid = np.repeat(np.arange(groups), per).astype(np.float64)
        feats = g.normal(size=(n, 2))
        u0 = g.normal(scale=1.2, size=groups)
        u1 = 0.5 * u0 + g.normal(scale=0.6, size=groups)
        y = 1.0 + feats @ np.array([2.0, -1.0]) + u0[gid.astype(int)] + 0.5 * g.normal(size=n)
        if slope:
            y = y + u1[gid.astype(int)] * feats[:, 0]
        x = np.column_stack([gid, feats])
        exog = np.column_stack([feats, np.ones(n)])
        exog_re = np.column_stack([np.ones(n), feats[:, 0]]) if slope else np.ones((n, 1))
        for reml in (True, False):
            model = MixedLM(y, exog, groups=gid, exog_re=exog_re)
            res = model.fit(reml=reml, method=["lbfgs"], gtol=1e-12, maxiter=5000)
            re = res.random_effects
            keys = sorted(re.keys())
            out.append(
                case(
                    f"{name}/{'reml' if reml else 'ml'}",
                    "mixed",
                    [x, y],
                    {
                        "coef": np.asarray(res.fe_params)[:2],
                        "intercept": np.array(float(np.asarray(res.fe_params)[2])),
                        "sigma2": np.array(float(res.scale)),
                        "randomCovariance": np.asarray(res.cov_re),
                        "logLikelihood": np.array(float(res.llf)),
                        "bse": np.asarray(res.bse_fe),
                        "randomEffects": np.array([np.asarray(re[k]) for k in keys]),
                    },
                    rtol=1e-3,
                    atol=1e-5,
                    reml=1.0 if reml else 0.0,
                    slope=1.0 if slope else 0.0,
                )
            )
    return out
