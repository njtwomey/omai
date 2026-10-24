"""Gaussian-process classification against scikit-learn: the Laplace approximation's marginal likelihood and
predictive probabilities at fixed hyperparameters, for two classes."""

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(111)
    x = g.normal(size=(40, 2))
    y = (x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * g.normal(size=40) > 0.4).astype(np.float64)
    q = g.normal(size=(6, 2))
    gpc = GaussianProcessClassifier(ConstantKernel(2.0) * RBF(0.8), optimizer=None).fit(x, y)  # pyright: ignore[reportArgumentType]
    return [
        case(
            "gpc/fixed",
            "gpc",
            [x, y, q],
            {"lml": np.array(gpc.log_marginal_likelihood_value_), "proba": gpc.predict_proba(q)},
            rtol=1e-7,
            atol=1e-8,
        )
    ]
