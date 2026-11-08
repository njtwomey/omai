"""Ridge regression in explicit kernel feature spaces against scikit-learn: Nystroem and RBFSampler pipelines with Ridge
(given the same landmarks and random draws), and the exact leave-one-out squared errors of RidgeCV, which omai's
kernel-selection criterion reproduces on the Gram matrix of the same features."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, uniform
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge, RidgeCV

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(77)
    x = uniform(g, (30, 2), -2.0, 2.0)
    y = np.sin(1.5 * x[:, 0]) + 0.5 * np.cos(2.0 * x[:, 1]) + 0.1 * g.normal(size=30)
    xt = uniform(g, (12, 2), -2.0, 2.0)
    out: list[Case] = []
    for gamma, m, alpha, fit_intercept in [(0.5, 10, 0.3, False), (1.0, 18, 0.05, True)]:
        ny = Nystroem(kernel="rbf", gamma=gamma, n_components=m, random_state=1).fit(x)
        ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(ny.transform(x), y)
        out.append(
            case(
                f"nystroem-{m}-{fit_intercept}",
                "nystroemRidge",
                [x, y, xt],
                {"predict": ridge.predict(ny.transform(xt)), "fitted": ridge.predict(ny.transform(x))},
                rtol=1e-8,
                atol=1e-9,
                gamma=gamma,
                indices=[int(i) for i in ny.component_indices_],
                alpha=alpha,
                fitIntercept=fit_intercept,
            )
        )
    for gamma, m, alpha in [(0.7, 40, 0.2)]:
        s = RBFSampler(gamma=gamma, n_components=m, random_state=3).fit(x)
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(s.transform(x), y)
        out.append(
            case(
                f"fourier-{m}",
                "fourierRidge",
                [x, y, xt, s.random_weights_, s.random_offset_],
                {"predict": ridge.predict(s.transform(xt))},
                rtol=1e-8,
                atol=1e-9,
                alpha=alpha,
            )
        )
    # Leave-one-out squared errors of the ridge on features z (no intercept): RidgeCV's efficient LOO.
    z = Nystroem(kernel="rbf", gamma=0.5, n_components=12, random_state=2).fit_transform(x)
    alphas = [0.01, 0.1, 1.0, 10.0]
    cv = RidgeCV(alphas=alphas, fit_intercept=False, store_cv_results=True, scoring=None).fit(z, y)
    out.append(
        case(
            "loo-ridgecv",
            "looRidgeCv",
            [z, y],
            {"mse": np.asarray(cv.cv_results_, dtype=np.float64).mean(axis=0)},
            rtol=1e-8,
            atol=1e-10,
            alphas=alphas,
        )
    )
    return out
