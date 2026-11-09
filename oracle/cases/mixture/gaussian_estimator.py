"""The GaussianMixture estimator against scikit-learn's, for every covariance type: from given initial weights, means
and precisions (so no random start is involved), EM to a tight tolerance, then the weights, means, covariances (as full
matrices), responsibilities, per-row log density, mean log likelihood, BIC, AIC and labels."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.mixture import GaussianMixture

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


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
    g = rng(31)
    centres = np.array([[-3.0, 0.0], [3.0, 1.0], [0.0, 4.0]])
    x = np.concatenate([c + g.normal(scale=[0.6, 0.9], size=(30, 2)) for c in centres])
    held = np.concatenate([c + g.normal(scale=[0.6, 0.9], size=(5, 2)) for c in centres])
    k, d = 3, 2
    weights0 = np.array([0.3, 0.3, 0.4])
    means0 = centres + g.normal(scale=0.5, size=(k, d))
    out: list[Case] = []
    for covariance_type in ["full", "tied", "diag", "spherical"]:
        # Initial covariances as full matrices (diagonal for diag and spherical), and the same as scikit-learn's precisions.
        cov0 = np.stack([np.diag([1.0 + 0.2 * j, 1.5 - 0.1 * j]) for j in range(k)])
        if covariance_type == "tied":
            cov0 = np.repeat(cov0[:1], k, axis=0)
        if covariance_type == "spherical":
            cov0 = np.stack([(1.0 + 0.2 * j) * np.eye(d) for j in range(k)])
        precisions: FloatArray
        if covariance_type == "full":
            precisions = np.linalg.inv(cov0)
        elif covariance_type == "tied":
            precisions = np.linalg.inv(cov0[0])
        elif covariance_type == "diag":
            precisions = np.stack([1.0 / np.diag(c) for c in cov0])
        else:
            precisions = np.array([1.0 / c[0, 0] for c in cov0])
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,  # pyright: ignore[reportArgumentType]
            weights_init=weights0,
            means_init=means0,
            precisions_init=precisions,
            tol=1e-10,
            max_iter=500,
            reg_covar=1e-6,
        ).fit(x)
        out.append(
            case(
                covariance_type,
                "mixture",
                [x, held, weights0, means0, cov0],
                {
                    "weights": model.weights_,
                    "means": model.means_,
                    "covariances": as_full(model.covariances_, covariance_type, k, d),
                    "predictProba": model.predict_proba(held),
                    "labels": model.predict(held).astype(np.float64),
                    "scoreSamples": model.score_samples(held),
                    "score": np.array(model.score(held)),
                    "bic": np.array(model.bic(held)),
                    "aic": np.array(model.aic(held)),
                },
                rtol=1e-6,
                atol=1e-8,
                covarianceType=covariance_type,
            )
        )
    return out
