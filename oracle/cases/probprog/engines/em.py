"""Expectation maximisation (probprog/engines/em.ts) against scikit-learn: `GaussianMixture` from the same initial
weights, means and variances (no covariance regularisation) reaches the same fixed point, and `FactorAnalysis` the same
maximum likelihood (the loadings are unique only up to sign, so the likelihood is compared)."""

import importlib.metadata

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.mixture import GaussianMixture
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": importlib.metadata.version("scikit-learn")}


def cases() -> list[Case]:
    g = rng(13)
    out: list[Case] = []
    datasets = {
        "separated": np.concatenate([g.normal(-2.0, 0.6, size=25), g.normal(1.5, 1.0, size=15)]),
        "overlapping": np.concatenate([g.normal(0.0, 1.0, size=30), g.normal(1.2, 0.7, size=30)]),
    }
    init_w = np.array([0.5, 0.5])
    init_mu = np.array([-1.0, 1.0])
    init_var = np.array([1.0, 1.0])
    for name, y in datasets.items():
        gm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            weights_init=init_w,
            means_init=init_mu[:, None],
            precisions_init=(1.0 / init_var)[:, None, None],
            reg_covar=0.0,
            tol=1e-14,
            max_iter=5000,
        )
        gm.fit(y[:, None])
        output = {
            "weights": np.asarray(gm.weights_, dtype=np.float64),
            "means": np.asarray(gm.means_, dtype=np.float64).ravel(),
            "variances": np.asarray(gm.covariances_, dtype=np.float64).ravel(),
        }
        out.append(
            case(
                f"gmm/{name}/logLikelihood",
                "gmmLogLikelihood",
                [y],
                np.array(float(gm.score(y[:, None])) * len(y)),
                rtol=1e-9,
            )
        )
        # EM converges linearly; where the components overlap the likelihood is so flat near its maximum that runs
        # stopping at the same likelihood (compared above at 1e-9) differ in the parameters' fourth digit.
        tol = 1e-6 if name == "separated" else 1e-3
        out.append(case(f"gmm/{name}", "gmm", [y], output, rtol=tol, atol=1e-8 if name == "separated" else 1e-4))
    # Strong loadings and little noise: EM converges fast when the factor is well determined.
    loadings = np.array([1.5, -1.0, 1.2])
    z = g.normal(size=12)
    x = z[:, None] * loadings + g.normal(size=(12, 3)) * np.array([0.3, 0.4, 0.25]) + np.array([1.0, 0.0, -1.0])
    fa = FactorAnalysis(n_components=1, tol=1e-12, max_iter=100000, svd_method="lapack")
    fa.fit(x)
    out.append(
        case("factorAnalysis/logLikelihood", "factorAnalysis", [x], np.array(float(fa.score(x)) * len(x)), rtol=1e-6)
    )
    return out
