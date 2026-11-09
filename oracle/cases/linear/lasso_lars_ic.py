"""The lasso with its penalty chosen by an information criterion against sklearn.linear_model.LassoLarsIC: the penalty,
coefficients, intercept, the criterion at every knot of the path and the noise variance, for the AIC and the BIC, with
the noise variance estimated by ordinary least squares or given, with and without an intercept."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.linear_model import LassoLarsIC

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(77)
    out: list[Case] = []
    for name, n, d, criterion, fit_intercept, noise_variance in [
        ("aic", 60, 8, "aic", True, None),
        ("bic", 60, 8, "bic", True, None),
        ("bic/no-intercept", 60, 8, "bic", False, None),
        ("aic/given-variance", 40, 12, "aic", True, 0.3),
        ("bic/wide-given-variance", 30, 45, "bic", True, 0.25),
    ]:
        x = g.normal(size=(n, d)) + 0.3
        x[:, 1] += 0.7 * x[:, 0]
        beta = np.zeros(d)
        beta[:3] = [3.0, -2.0, 1.5]
        y = x @ beta + 1.0 + 0.5 * g.normal(size=n)
        held = g.normal(size=(6, d)) + 0.3
        model = LassoLarsIC(criterion=criterion, fit_intercept=fit_intercept, noise_variance=noise_variance).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                name,
                "ic",
                [x, y, held],
                {
                    "coef": model.coef_,
                    "intercept": np.array(float(model.intercept_)),
                    "alpha": np.array(float(model.alpha_)),
                    "criterion": model.criterion_,
                    "noiseVariance": np.array(float(model.noise_variance_)),
                    "predict": model.predict(held),
                },
                rtol=1e-7,
                atol=1e-9,
                criterion=criterion,
                fitIntercept=fit_intercept,
                noiseVariance=-1.0 if noise_variance is None else noise_variance,
            )
        )
    return out
