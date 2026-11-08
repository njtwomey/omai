"""Bayesian linear models and isotonic regression against scikit-learn: BayesianRidge (coefficients, precisions,
posterior covariance, predictive standard deviations), ARDRegression and IsotonicRegression."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ARDRegression, BayesianRidge

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(31)
    n = 50
    x = g.normal(size=(n, 4))
    y = 0.5 + x @ np.array([1.0, 0.0, -0.8, 0.0]) + 0.4 * g.normal(size=n)
    out: list[Case] = []
    br = BayesianRidge().fit(x, y)
    _, std = br.predict(x[:5], return_std=True)
    out.append(
        case(
            "bayesianRidge",
            "bayesianRidge",
            [x, y],
            {
                "coef": br.coef_,
                "intercept": np.array(br.intercept_),
                "alpha": np.array(br.alpha_),
                "lambda": np.array(br.lambda_),
                "sigma": br.sigma_,
                "std": std,
            },
            rtol=1e-9,
            atol=1e-10,
        )
    )
    ard = ARDRegression().fit(x, y)
    out.append(
        case(
            "ard",
            "ard",
            [x, y],
            {"coef": ard.coef_, "intercept": np.array(ard.intercept_), "alpha": np.array(ard.alpha_)},
            rtol=1e-7,
            atol=1e-8,
        )
    )
    xi = np.round(g.uniform(0, 10, size=30), 1)
    yi = np.sqrt(xi) + 0.5 * g.normal(size=30)
    test = np.linspace(-1, 11, 13)
    for inc in [True, False]:
        iso = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(xi, yi if inc else -yi)
        out.append(
            case(
                f"isotonic/{inc}",
                "isotonic",
                [xi, yi if inc else -yi, test],
                {"predict": iso.predict(test)},
                increasing=inc,
            )
        )
    return out
