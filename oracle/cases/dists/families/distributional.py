"""Distributional (GAMLSS) and ordinal families (dists/families/distributional.ts) against scipy.stats."""

import numpy as np
import scipy
from scipy import stats
from scipy.special import expit
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, Unary, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def ordinal_loglike(y: FloatArray, eta: FloatArray, cuts: FloatArray, cdf: Unary) -> float:
    """sum_i log(F(c_{y_i} - eta_i) - F(c_{y_i - 1} - eta_i)) with c_{-1} = -inf and c_{K-1} = inf."""
    full = np.concatenate([[-np.inf], cuts, [np.inf]])
    k = y.astype(int)
    return float(np.log(cdf(full[k + 1] - eta) - cdf(full[k] - eta)).sum())


def cases() -> list[Case]:
    g = rng(50)
    out: list[Case] = []
    n = 6
    pos = np.exp(g.normal(size=n))
    unit = g.uniform(0.05, 0.95, n)
    counts = np.array([0.0, 1, 3, 2, 0, 6])
    real = g.normal(size=n) * 2
    eta = [g.normal(size=n) * 0.3 for _ in range(3)]
    ident, log_inv = eta, [np.exp(e) for e in eta]
    families: list[tuple[str, FloatArray, list[FloatArray], float]] = [
        ("normal", real, [eta[0], eta[1]], float(stats.norm(ident[0], log_inv[1]).logpdf(real).sum())),
        (
            "studentT",
            real,
            [eta[0], eta[1], eta[2]],
            float(stats.t(log_inv[2], ident[0], log_inv[1]).logpdf(real).sum()),
        ),
        (
            "skewNormal",
            real,
            [eta[0], eta[1], eta[2]],
            float(stats.skewnorm(ident[2], ident[0], log_inv[1]).logpdf(real).sum()),
        ),
        (
            "gamma",
            pos,
            [eta[0], eta[1]],
            float(stats.gamma(1 / log_inv[1] ** 2, scale=log_inv[0] * log_inv[1] ** 2).logpdf(pos).sum()),
        ),
        (
            "logNormal",
            pos,
            [eta[0], eta[1]],
            float(stats.lognorm(log_inv[1], scale=np.exp(ident[0])).logpdf(pos).sum()),
        ),
        ("weibull", pos, [eta[0], eta[1]], float(stats.weibull_min(log_inv[1], scale=log_inv[0]).logpdf(pos).sum())),
        ("poisson", counts, [eta[0]], float(stats.poisson(log_inv[0]).logpmf(counts).sum())),
        (
            "negativeBinomial",
            counts,
            [eta[0], eta[1]],
            float(stats.nbinom(1 / log_inv[1], 1 / (1 + log_inv[1] * log_inv[0])).logpmf(counts).sum()),
        ),
    ]
    mu_b, s_b = expit(eta[0]), expit(eta[1])
    k = (1 - s_b**2) / s_b**2
    families.append(("beta", unit, [eta[0], eta[1]], float(stats.beta(mu_b * k, (1 - mu_b) * k).logpdf(unit).sum())))
    zi_mu, zi_gate = log_inv[0], expit(eta[1])
    zip_ll = np.where(
        counts == 0,
        np.log(zi_gate + (1 - zi_gate) * np.exp(-zi_mu)),
        np.log(1 - zi_gate) + stats.poisson(zi_mu).logpmf(counts),
    )
    families.append(("zeroInflatedPoisson", counts, [eta[0], eta[1]], float(zip_ll.sum())))
    for name, y, predictors, ll in families:
        out.append(case(f"gamlss/{name}", f"gamlss.{name}", [y, *predictors], np.array(ll), rtol=1e-10))
    cuts = np.array([-1.0, 0.2, 1.4])
    y = np.array([0.0, 1, 3, 2, 2, 0, 1, 3])
    pred = g.normal(size=8)
    cdfs: dict[str, Unary] = {
        "logit": expit,
        "probit": stats.norm.cdf,
        "cloglog": stats.gumbel_l.cdf,
        "loglog": stats.gumbel_r.cdf,
        "cauchit": stats.cauchy.cdf,
    }
    for name, cdf in cdfs.items():
        out.append(
            case(
                f"ordinal/{name}",
                f"ordinal.{name}",
                [y, pred, cuts],
                np.array(ordinal_loglike(y, pred, cuts, cdf)),
                rtol=1e-10,
            )
        )
    return out
