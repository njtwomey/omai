"""Censored (Tobit) regression against statsmodels: the Gaussian censored likelihood written as a
GenericLikelihoodModel (statsmodels has no Tobit model) and fitted by Newton from the least-squares start; compared on
the coefficients, the noise scale, the log-likelihood and the Wald standard errors (numerical Hessian)."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}

type Vec = npt.NDArray[np.float64]


class Tobit(GenericLikelihoodModel):  # type: ignore[misc]
    def __init__(self, endog: Vec, exog: Vec, lower: float, upper: float) -> None:
        super().__init__(endog, exog)  # pyright: ignore[reportUnknownMemberType]
        self.lower = lower
        self.upper = upper

    def nloglikeobs(self, params: Vec) -> Vec:  # pyright: ignore[reportIncompatibleMethodOverride]
        beta, sigma = params[:-1], float(np.exp(params[-1]))
        y = np.asarray(self.endog, dtype=np.float64)
        eta = np.asarray(self.exog, dtype=np.float64) @ beta
        ll = norm.logpdf((y - eta) / sigma) - np.log(sigma)
        low = y <= self.lower
        high = y >= self.upper
        ll = np.where(low, norm.logcdf((self.lower - eta) / sigma), ll)
        ll = np.where(high, norm.logsf((self.upper - eta) / sigma), ll)
        return -np.asarray(ll, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(1958)
    out: list[Case] = []
    for name, n, d, lower, upper in [
        ("left", 300, 2, 0.0, np.inf),
        ("both", 400, 1, -0.5, 3.0),
        ("right", 250, 3, -np.inf, 2.0),
    ]:
        x = g.normal(size=(n, d))
        latent = x @ np.linspace(1.2, -0.6, d) + 0.8 + 1.1 * g.normal(size=n)
        y = np.clip(latent, lower, upper)
        exog = np.column_stack([x, np.ones(n)])
        ols = np.linalg.lstsq(exog, y, rcond=None)[0]
        start = np.append(ols, np.log(np.std(y - exog @ ols)))
        model = Tobit(y, exog, lower, upper)
        res = model.fit(start_params=start, method="newton", maxiter=200, disp=0, gtol=1e-12)  # pyright: ignore[reportUnknownMemberType]
        params = np.asarray(res.params, dtype=np.float64)
        bse = np.asarray(res.bse, dtype=np.float64)
        sigma = float(np.exp(params[-1]))
        out.append(
            case(
                f"tobit/{name}",
                "tobit",
                [x, y],
                {
                    "coef": params[:d],
                    "intercept": np.array(float(params[d])),
                    "sigma": np.array(sigma),
                    "logLikelihood": np.array(float(cast(Any, res).llf)),
                    "bse": np.append(bse[: d + 1], sigma * bse[-1]),
                },
                rtol=1e-4,
                atol=1e-5,
                lower=float(lower) if np.isfinite(lower) else -1e300,
                upper=float(upper) if np.isfinite(upper) else 1e300,
            )
        )
    return out
