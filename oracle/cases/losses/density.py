"""Likelihood losses against torch.distributions, torch.nn.functional and scipy.stats (float64)."""

import numpy as np
import scipy
import scipy.stats as st
import torch
import torch.distributions as D
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(43)
    loc, scale = g.normal(size=8), np.exp(0.3 * g.normal(size=8))
    x = g.normal(size=8)
    mean, logvar, y = g.normal(size=(4, 3)), 0.5 * g.normal(size=(4, 3)), g.normal(size=(4, 3))
    out: list[Case] = [
        case(
            "nll/normal",
            "nllNormal",
            [loc, scale, x],
            to_numpy(-D.Normal(to_torch(loc), to_torch(scale)).log_prob(to_torch(x)).mean()),
        ),
        case(
            "heteroscedastic",
            "heteroscedasticGaussianNll",
            [mean, logvar, y],
            to_numpy(F.gaussian_nll_loss(to_torch(mean), to_torch(y), to_torch(np.exp(logvar)), eps=0.0, full=True)),
            full=True,
        ),
        case(
            "heteroscedastic/beta",
            "heteroscedasticGaussianNll",
            [mean, logvar, y],
            np.asarray((0.5 * (logvar + (y - mean) ** 2 / np.exp(logvar)) * np.exp(logvar) ** 0.5).sum()),
            beta=0.5,
            reduction="sum",
        ),
    ]
    # Mixture density networks: univariate and diagonal multivariate components.
    n, k, d = 5, 3, 2
    logits = g.normal(size=(n, k))
    mu1, s1, t1 = g.normal(size=(n, k)), np.exp(0.4 * g.normal(size=(n, k))), g.normal(size=n)
    mix1 = D.MixtureSameFamily(D.Categorical(logits=to_torch(logits)), D.Normal(to_torch(mu1), to_torch(s1)))
    out.append(
        case(
            "mdn/univariate",
            "mixtureDensityNll",
            [logits, mu1, s1, t1],
            to_numpy(-mix1.log_prob(to_torch(t1))),
            reduction="none",
        )
    )
    mu2, s2, t2 = g.normal(size=(n, k, d)), np.exp(0.4 * g.normal(size=(n, k, d))), g.normal(size=(n, d))
    comp = D.Independent(D.Normal(to_torch(mu2), to_torch(s2)), 1)
    mix2 = D.MixtureSameFamily(D.Categorical(logits=to_torch(logits)), comp)
    out.append(
        case("mdn/diagonal", "mixtureDensityNll", [logits, mu2, s2, t2], to_numpy(-mix2.log_prob(to_torch(t2)).mean()))
    )
    # Censored observations, two of each kind: exact, right-censored, left-censored, interval.
    lo = np.array([0.3, -1.2, 0.5, 1.1, -np.inf, -np.inf, -0.4, 1.5])
    hi = np.array([0.3, -1.2, np.inf, np.inf, 0.2, -1.0, 0.9, 2.5])
    norm = st.norm(loc, scale)
    loglik = np.concatenate(
        [
            norm.logpdf(lo)[:2],
            norm.logsf(lo)[2:4],
            norm.logcdf(hi)[4:6],
            np.log((norm.cdf(hi) - norm.cdf(lo))[6:]),
        ]
    )
    out.append(case("censored/normal", "censoredNormal", [loc, scale, lo, hi], -loglik, reduction="none"))
    shape = np.exp(0.3 * g.normal(size=8)) + 0.5
    wlo = np.array([0.3, 1.2, 0.5, 1.1, -np.inf, -np.inf, 0.4, 1.5])
    whi = np.array([0.3, 1.2, np.inf, np.inf, 0.2, 1.0, 0.9, 2.5])
    wb = st.weibull_min(shape, scale=scale)
    wloglik = np.concatenate(
        [
            wb.logpdf(wlo)[:2],
            wb.logsf(wlo)[2:4],
            wb.logcdf(whi)[4:6],
            np.log((wb.cdf(whi) - wb.cdf(wlo))[6:]),
        ]
    )
    out.append(
        case("censored/weibull", "censoredWeibull", [shape, scale, wlo, whi], np.asarray(-wloglik.mean()), rtol=1e-10)
    )
    return out
