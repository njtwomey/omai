"""Ordinal regression against statsmodels' OrderedModel (cumulative links logit, probit, cloglog and loglog): slopes,
thresholds, log-likelihood, slope standard errors and the category probabilities."""

import numpy as np
import scipy.stats as st
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.miscmodels.ordinal_model import OrderedModel

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(1951)
    out: list[Case] = []
    links = {"logit": "logit", "probit": "probit", "cloglog": st.gumbel_l, "loglog": st.gumbel_r}
    for name, n, d, k in [("three", 120, 2, 3), ("five", 200, 3, 5)]:
        x = g.normal(size=(n, d))
        latent = x @ np.linspace(1.0, -0.7, d) + g.logistic(size=n)
        cuts = np.quantile(latent, np.linspace(0, 1, k + 1)[1:-1])
        y = np.searchsorted(cuts, latent).astype(np.float64)
        for link, distr in links.items():
            res = OrderedModel(y, x, distr=distr).fit(  # pyright: ignore[reportArgumentType]
                method="bfgs", disp=False, gtol=1e-10, maxiter=5000
            )
            thresholds = np.asarray(res.model.transform_threshold_params(res.params))[1:-1]
            out.append(
                case(
                    f"cumulative/{name}/{link}",
                    "cumulative",
                    [x, y],
                    {
                        "coef": np.asarray(res.params)[:d],
                        "thresholds": thresholds,
                        "logLikelihood": np.array(float(res.llf)),
                        "bse": np.asarray(res.bse)[:d],
                        "proba": np.asarray(res.model.predict(res.params, exog=x[:10])),
                    },
                    rtol=1e-5,
                    atol=1e-6,
                    link=link,
                )
            )
    return out
