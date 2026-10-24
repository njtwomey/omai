"""GLM regressors against scikit-learn (penalised Poisson, gamma and Tweedie) and statsmodels (unpenalised Poisson with
Wald standard errors)."""

import numpy as np
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(11)
    n = 80
    x = g.normal(size=(n, 3)) * 0.6
    mu = np.exp(0.4 + x @ np.array([0.5, -0.3, 0.2]))
    counts = g.poisson(mu).astype(np.float64)
    positive = g.gamma(2.0, mu / 2.0)
    out: list[Case] = []
    for name, model, y in [
        ("poisson", PoissonRegressor(alpha=0.5, tol=1e-12, max_iter=10000, solver="newton-cholesky"), counts),
        ("gamma", GammaRegressor(alpha=0.3, tol=1e-12, max_iter=10000, solver="newton-cholesky"), positive),
        (
            "tweedie",
            TweedieRegressor(power=1.5, alpha=0.2, link="log", tol=1e-12, max_iter=10000, solver="newton-cholesky"),
            positive,
        ),
    ]:
        model.fit(x, y)
        out.append(
            case(
                name,
                name,
                [x, y],
                {"coef": model.coef_, "intercept": np.array(model.intercept_)},
                rtol=1e-7,
                atol=1e-9,
                alpha=float(model.alpha),
            )
        )
    glm = sm.GLM(counts, np.column_stack([x, np.ones(n)]), family=sm.families.Poisson()).fit(tol=1e-13)
    out.append(
        case("poisson/wald", "poissonWald", [x, counts], {"params": glm.params, "bse": glm.bse}, rtol=1e-8, atol=1e-10)
    )
    return out
