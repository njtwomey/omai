"""Canonical-form Gaussian factors against numpy and scipy.stats: marginals and conditionals of a random joint Gaussian
(Schur complements in moment form), products of two densities with their normalising constant, and log densities."""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd
from scipy.stats import multivariate_normal

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

NAMES = ["a", "b", "c", "d", "e"]


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(17)
    for t in range(3):
        mean = g.normal(size=5)
        cov = spd(g, 5) / 5
        keep = [0, 2, 3] if t == 0 else [1, 4] if t == 1 else [4, 0, 2]
        drop = [i for i in range(5) if i not in keep]
        # Marginal: the sub-mean and sub-covariance.
        out.append(
            case(
                f"marginal/{t}",
                "marginal",
                [mean, cov],
                np.concatenate([mean[keep], cov[np.ix_(keep, keep)].ravel()]),
                rtol=1e-10,
                atol=1e-12,
                keep=[NAMES[i] for i in keep],
            )
        )
        # Conditional on the dropped variables at values y: mean_k + S_kd S_dd^{-1} (y - mean_d), Schur complement.
        y = g.normal(size=len(drop))
        s_kd = cov[np.ix_(keep, drop)]
        s_dd = cov[np.ix_(drop, drop)]
        cond_mean = mean[keep] + s_kd @ np.linalg.solve(s_dd, y - mean[drop])
        cond_cov = cov[np.ix_(keep, keep)] - s_kd @ np.linalg.solve(s_dd, s_kd.T)
        log_evidence = multivariate_normal(mean[drop], s_dd).logpdf(y)
        out.append(
            case(
                f"conditional/{t}",
                "conditional",
                [mean, cov, y],
                np.concatenate([cond_mean, cond_cov.ravel(), [log_evidence]]),
                rtol=1e-10,
                atol=1e-12,
                keep=[NAMES[i] for i in keep],
                observed=[NAMES[i] for i in drop],
            )
        )
        # The product of two densities over the same variables: a Gaussian times N(mean1; mean2, cov1 + cov2).
        mean2 = g.normal(size=5)
        cov2 = spd(g, 5) / 3
        p1 = np.linalg.inv(cov)
        p2 = np.linalg.inv(cov2)
        prod_cov: FloatArray = np.linalg.inv(p1 + p2)
        prod_mean = prod_cov @ (p1 @ mean + p2 @ mean2)
        log_mass = multivariate_normal(mean2, cov + cov2).logpdf(mean)
        out.append(
            case(
                f"product/{t}",
                "product",
                [mean, cov, mean2, cov2],
                np.concatenate([prod_mean, prod_cov.ravel(), [log_mass]]),
                rtol=1e-10,
                atol=1e-12,
            )
        )
        points = g.normal(size=(4, 5))
        out.append(
            case(
                f"logpdf/{t}",
                "logpdf",
                [mean, cov, points],
                np.asarray(multivariate_normal(mean, cov).logpdf(points)),
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
