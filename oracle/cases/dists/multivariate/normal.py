"""Multivariate normal families (dists/multivariate/normal.ts) against scipy.stats and torch.distributions."""

import numpy as np
import scipy
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, spd
from omai_oracle.torch_helpers import to_numpy, to_torch
from scipy import stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(32)
    out: list[Case] = []
    mean = np.array([0.5, -1.0, 2.0])
    cov = spd(g, 3) / 3
    x = g.normal(size=(6, 3))
    mvn = stats.multivariate_normal(mean, cov)
    out.append(case("mvn/logProb", "multivariateNormal.logProb", [mean, cov, x], mvn.logpdf(x), rtol=1e-10))
    out.append(case("mvn/entropy", "multivariateNormal.entropy", [mean, cov], np.array(mvn.entropy()), rtol=1e-12))
    out.append(case("mvn/variance", "multivariateNormal.variance", [mean, cov], np.diag(cov), rtol=1e-12))
    # Conditioning on coordinate 2 = 1.5, against the textbook Gaussian conditioning formulas.
    a, b = [0, 1], [2]
    s_ab, s_bb = cov[np.ix_(a, b)], cov[np.ix_(b, b)]
    cond_mean = mean[a] + (s_ab @ np.linalg.solve(s_bb, np.array([1.5]) - mean[b]))
    cond_cov = cov[np.ix_(a, a)] - s_ab @ np.linalg.solve(s_bb, s_ab.T)
    out.append(case("mvn/condition/mean", "multivariateNormal.conditionMean", [mean, cov], cond_mean, rtol=1e-10))
    out.append(case("mvn/condition/cov", "multivariateNormal.conditionCov", [mean, cov], cond_cov, rtol=1e-10))
    # A batch of covariances against a batch of points.
    covs = np.stack([spd(g, 3) / 3, spd(g, 3) / 2])
    means = g.normal(size=(2, 3))
    xb = g.normal(size=(4, 2, 3))
    batch = np.stack([stats.multivariate_normal(means[i], covs[i]).logpdf(xb[:, i]) for i in range(2)], axis=-1)
    out.append(case("mvn/batch/logProb", "multivariateNormal.logProb", [means, covs, xb], batch, rtol=1e-10))
    # Low-rank plus diagonal covariance against torch.distributions.LowRankMultivariateNormal.
    factor = g.normal(size=(4, 2))
    diag = np.exp(g.normal(size=4))
    loc = g.normal(size=4)
    xs = g.normal(size=(5, 4))
    lr = torch.distributions.LowRankMultivariateNormal(to_torch(loc), to_torch(factor), to_torch(diag))
    params = [loc, factor, diag]
    out.append(
        case(
            "lowRank/logProb",
            "lowRankMultivariateNormal.logProb",
            [*params, xs],
            to_numpy(lr.log_prob(to_torch(xs))),
            rtol=1e-10,
        )
    )
    out.append(case("lowRank/entropy", "lowRankMultivariateNormal.entropy", params, to_numpy(lr.entropy()), rtol=1e-10))
    out.append(
        case("lowRank/variance", "lowRankMultivariateNormal.variance", params, to_numpy(lr.variance), rtol=1e-12)
    )
    # Matrix normal against scipy.stats.matrix_normal.
    m = g.normal(size=(3, 2))
    u = spd(g, 3) / 3
    v = spd(g, 2) / 2
    xm = g.normal(size=(4, 3, 2))
    mn = stats.matrix_normal(m, u, v)
    out.append(case("matrixNormal/logProb", "matrixNormal.logProb", [m, u, v, xm], mn.logpdf(xm), rtol=1e-10))
    out.append(case("matrixNormal/entropy", "matrixNormal.entropy", [m, u, v], np.array(mn.entropy()), rtol=1e-10))
    return out
