"""Mixtures (dists/compose/mixture.ts) against scipy.stats.Mixture and torch.distributions.MixtureSameFamily."""

import numpy as np
import scipy
import torch
from omai_oracle.distributions import method_case
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(43)
    q = np.array([0.05, 0.3, 0.5, 0.8, 0.95])
    out: list[Case] = []
    # A normal-gamma mixture on the real line: weights, then the normal's and the gamma's parameters.
    w = np.array([0.3, 0.7])
    params = [w, np.array([0.0]), np.array([1.0]), np.array([3.0]), np.array([2.0])]
    gamma = stats.make_distribution(stats.gamma)
    rv = stats.Mixture([stats.Normal(mu=0.0, sigma=1.0), gamma(a=3.0) / 2.0], weights=w)
    x = g.uniform(0.05, 4.0, 7)
    label = "normalGammaMixture"
    out += [
        method_case(label, label, "logProb", params, x, rv.logpdf(x), rtol=1e-12),
        method_case(label, label, "cdf", params, x, rv.cdf(x), rtol=1e-12),
        method_case(label, label, "sf", params, x, rv.ccdf(x), rtol=1e-12),
        method_case(label, label, "ppf", params, q, rv.icdf(q), rtol=1e-9),
        method_case(label, label, "mean", params, None, np.atleast_1d(rv.mean()), rtol=1e-12),
        method_case(label, label, "variance", params, None, np.atleast_1d(rv.variance()), rtol=1e-12),
        method_case(label, label, "entropy", params, None, np.atleast_1d(rv.entropy()), rtol=1e-8),
    ]
    # A mixture of three bivariate normals against torch's MixtureSameFamily.
    probs = np.array([0.2, 0.5, 0.3])
    locs = g.normal(size=(3, 2))
    m = g.normal(size=(3, 2, 2))
    covs = m @ np.swapaxes(m, -1, -2) + np.eye(2)
    xs = g.normal(size=(5, 2))
    mix = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=to_torch(probs)),
        torch.distributions.MultivariateNormal(to_torch(locs), covariance_matrix=to_torch(covs)),
    )
    inputs = [probs, locs, covs]
    out += [
        case(
            "mvnMixture/logProb", "mvnMixture.logProb", [*inputs, xs], to_numpy(mix.log_prob(to_torch(xs))), rtol=1e-10
        ),
        case("mvnMixture/mean", "mvnMixture.mean", inputs, to_numpy(mix.mean)),
        case("mvnMixture/variance", "mvnMixture.variance", inputs, to_numpy(mix.variance)),
    ]
    return out
