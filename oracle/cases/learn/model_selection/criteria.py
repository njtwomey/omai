"""Model-selection criteria against statsmodels (OLS log-likelihood, AIC, BIC and the corrected AIC) and the exact
leave-one-out residuals of least squares against sklearn's refitting with LeaveOneOut."""

import numpy as np
import sklearn
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from statsmodels.tools.eval_measures import aicc

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(4339)
    n = 25
    x = g.normal(size=(n, 3))
    y = x @ np.array([0.5, -1.0, 2.0]) + 1.0 + 0.4 * g.normal(size=n)
    out: list[Case] = []
    for p in [1, 2, 3]:
        design = sm.add_constant(x[:, :p])
        fit = sm.OLS(y, design).fit()
        k = p + 1
        out.append(
            case(
                f"ols/{p}",
                "ols",
                [x[:, :p], y],
                {
                    "rss": np.array(fit.ssr),
                    "llf": np.array(fit.llf),
                    "aic": np.array(fit.aic),
                    "bic": np.array(fit.bic),
                    "aicc": np.array(aicc(fit.llf, n, k)),
                },
                k=k,
                rtol=1e-10,
                atol=1e-12,
            )
        )
    loo = cross_val_predict(LinearRegression(), x, y, cv=LeaveOneOut())
    out.append(case("loo", "loo", [x, y], {"residuals": y - loo}, rtol=1e-9, atol=1e-12))
    return out
