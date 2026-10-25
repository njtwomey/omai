"""Structural combinators (dists/compose/structure.ts) against scipy.stats and torch.distributions."""

import numpy as np
import scipy
import torch
from mdml_oracle.distributions import method_case
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from scipy import stats

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(42)
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out: list[Case] = []
    # Truncations through scipy's random-variable infrastructure (stats.truncate).
    truncations = [
        (
            "truncatedGamma",
            [np.array([2.0]), np.array([1.5]), np.array([0.5]), np.array([3.0])],
            stats.truncate(stats.make_distribution(stats.gamma)(a=2.0) / 1.5, 0.5, 3.0),
            g.uniform(0.5, 3.0, 6),
        ),
        (
            "truncatedNormalAbove",
            [np.array([1.0]), np.array([2.0]), np.array([2.0]), np.array([np.inf])],
            stats.truncate(stats.Normal(mu=1.0, sigma=2.0), 2.0, np.inf),
            2.0 + np.exp(g.normal(size=6)),
        ),
        (
            "truncatedNormalTail",
            [np.array([0.0]), np.array([1.0]), np.array([5.0]), np.array([6.0])],
            stats.truncate(stats.Normal(mu=0.0, sigma=1.0), 5.0, 6.0),
            g.uniform(5.0, 6.0, 6),
        ),
    ]
    for label, params, rv, x in truncations:
        out += [
            method_case(label, label, "logProb", params, x, rv.logpdf(x), rtol=1e-9),
            method_case(label, label, "cdf", params, x, rv.cdf(x), rtol=1e-9),
            method_case(label, label, "sf", params, x, rv.ccdf(x), rtol=1e-9),
            method_case(label, label, "ppf", params, q, rv.icdf(q), rtol=1e-9),
            method_case(label, label, "mean", params, None, np.atleast_1d(rv.mean()), rtol=1e-8),
            method_case(label, label, "variance", params, None, np.atleast_1d(rv.variance()), rtol=1e-8),
            method_case(label, label, "entropy", params, None, np.atleast_1d(rv.entropy()), rtol=1e-8),
        ]
    # Zero-truncated Poisson against its definition from scipy's pmf.
    lam = 3.0
    ks = np.arange(1.0, 9.0)
    pois = stats.poisson(lam)
    zt = pois.pmf(ks) / pois.sf(0)
    params = [np.array([lam])]
    out += [
        method_case("zeroTruncatedPoisson", "zeroTruncatedPoisson", "logProb", params, ks, np.log(zt), rtol=1e-12),
        method_case(
            "zeroTruncatedPoisson",
            "zeroTruncatedPoisson",
            "mean",
            params,
            None,
            np.atleast_1d(lam / pois.sf(0)),
            rtol=1e-12,
        ),
    ]
    # exp then affine of a normal against torch's TransformedDistribution.
    base = torch.distributions.Normal(to_torch(np.array([0.5])), to_torch(np.array([0.8])))
    flow = torch.distributions.TransformedDistribution(
        base, [torch.distributions.ExpTransform(), torch.distributions.AffineTransform(1.0, 2.0)]
    )
    y = 1.0 + np.exp(g.normal(size=6))
    out.append(
        method_case(
            "expAffineNormal",
            "expAffineNormal",
            "logProb",
            [np.array([0.5]), np.array([0.8])],
            y,
            to_numpy(flow.log_prob(to_torch(y))),
            rtol=1e-12,
        )
    )
    # The log-normal through the transform: moments and entropy against scipy.
    ln = stats.lognorm(0.8, scale=np.exp(0.5))
    params = [np.array([0.5]), np.array([0.8])]
    out += [
        method_case("expNormal", "expNormal", "mean", params, None, np.atleast_1d(ln.mean()), rtol=1e-9),
        method_case("expNormal", "expNormal", "variance", params, None, np.atleast_1d(ln.var()), rtol=1e-8),
        method_case("expNormal", "expNormal", "entropy", params, None, np.atleast_1d(ln.entropy()), rtol=1e-9),
    ]
    # The empirical distribution against numpy.
    atoms = np.round(g.normal(size=12), 1)
    weights = g.uniform(0.5, 2.0, 12)
    w = weights / weights.sum()
    mean = float((w * atoms).sum())
    out += [
        method_case("empirical", "empirical", "mean", [atoms, weights], None, np.array(mean)),
        method_case(
            "empirical", "empirical", "variance", [atoms, weights], None, np.array((w * (atoms - mean) ** 2).sum())
        ),
        method_case(
            "empirical",
            "empirical",
            "ppf",
            [atoms, weights],
            q,
            np.quantile(atoms, q, weights=weights, method="inverted_cdf"),
        ),
    ]
    return out
