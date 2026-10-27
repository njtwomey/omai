"""The Bayesian Gaussian mixture by variational inference against scikit-learn's BayesianGaussianMixture, for a
Dirichlet-distribution and a Dirichlet-process weight prior and every covariance type. scikit-learn starts from random
or k-means responsibilities drawn inside `fit`; here it starts from given responsibilities (its private `_initialize`),
so the engine can run the same coordinate ascent for a fixed number of iterations and match weights, means,
covariances (as full matrices), degrees of freedom, mean precisions, concentrations, lower bound and predicted
responsibilities."""

import warnings

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


class FromResponsibilities(BayesianGaussianMixture):
    """The variational mixture started from given responsibilities instead of its random or k-means start."""

    resp0: FloatArray

    def _initialize_parameters(self, X: FloatArray, random_state: object, xp: object = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self._initialize(X, self.resp0)  # pyright: ignore[reportPrivateUsage]


def as_full(covariances: FloatArray, covariance_type: str, k: int, d: int) -> FloatArray:
    """scikit-learn's per-type covariance shapes as full matrices `[k, d, d]`."""
    if covariance_type == "full":
        return covariances
    if covariance_type == "tied":
        return np.repeat(covariances[None], k, axis=0)
    if covariance_type == "diag":
        return np.stack([np.diag(c) for c in covariances])
    return np.stack([c * np.eye(d) for c in covariances])


def cases() -> list[Case]:
    g = rng(97)
    centres = np.array([[-3.0, 0.0], [3.0, 1.0], [0.0, 4.0]])
    x = np.concatenate([c + g.normal(scale=[0.6, 0.9], size=(25, 2)) for c in centres])
    k = 5
    resp = g.uniform(size=(x.shape[0], k))
    resp /= resp.sum(axis=1, keepdims=True)
    out: list[Case] = []
    for prior_type, concentration in [("dirichlet_process", 0.05), ("dirichlet_distribution", 0.4)]:
        for covariance_type in ["full", "tied", "diag", "spherical"]:
            for iterations in [1, 6, 40]:
                if iterations == 40 and covariance_type != "full":
                    continue
                model = FromResponsibilities(
                    n_components=k,
                    covariance_type=covariance_type,  # pyright: ignore[reportArgumentType]
                    weight_concentration_prior_type=prior_type,
                    weight_concentration_prior=concentration,
                    max_iter=iterations,
                    tol=0.0,
                    random_state=0,
                )
                model.resp0 = resp
                with warnings.catch_warnings():
                    # A fixed number of iterations is the point: the run is not meant to converge.
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(x)
                post = (
                    np.stack(model.weight_concentration_)
                    if prior_type == "dirichlet_process"
                    else model.weight_concentration_
                )
                out.append(
                    case(
                        f"{prior_type}/{covariance_type}/{iterations}",
                        "variational",
                        [x, resp],
                        {
                            "weights": model.weights_,
                            "means": model.means_,
                            "covariances": as_full(model.covariances_, covariance_type, k, x.shape[1]),
                            "degreesOfFreedom": np.broadcast_to(model.degrees_of_freedom_, (k,)).copy(),
                            "meanPrecision": model.mean_precision_,
                            "concentration": np.asarray(post),
                            "lowerBound": np.array(float(np.ravel(model.lower_bound_)[0])),
                            "predictProba": model.predict_proba(x),
                        },
                        rtol=1e-7,
                        atol=1e-9,
                        covarianceType=covariance_type,
                        weightPrior=prior_type,
                        concentration=concentration,
                        iterations=iterations,
                    )
                )
    return out
