"""Zero-inflated and hurdle count regression against statsmodels: ZeroInflatedPoisson and ZeroInflatedNegativeBinomialP
(logit inflation, constant-only and with a regressor), and the hurdle model as its two separate parts, a Logit for
whether the count is zero and the zero-truncated Poisson or negative binomial (TruncatedLFPoisson and
TruncatedLFNegativeBinomialP) for the positive counts. Coefficients, dispersion and log-likelihood."""

import numpy as np
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP, TruncatedLFPoisson

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def draw(g: np.random.Generator, n: int, d: int, family: str) -> tuple[np.ndarray, np.ndarray]:
    x = g.normal(size=(n, d))
    mu = np.exp(0.6 + 0.5 * x[:, 0] - 0.3 * x[:, -1])
    counts = g.poisson(mu) if family == "poisson" else g.poisson(mu * g.gamma(2.0, 0.5, size=n))
    zero = g.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.4 + 0.8 * x[:, 0])))
    return x, np.where(zero, 0, counts).astype(np.float64)


def cases() -> list[Case]:
    g = rng(1992)
    out: list[Case] = []
    for family in ("poisson", "negative-binomial"):
        for name, n, d, infl in [("constant", 400, 2, False), ("regressor", 500, 2, True)]:
            x, y = draw(g, n, d, family)
            exog = np.column_stack([x, np.ones(n)])
            zfeat = [0] if infl else []
            exog_infl = np.column_stack([x[:, zfeat], np.ones(n)])
            if family == "poisson":
                zi = ZeroInflatedPoisson(y, exog, exog_infl=exog_infl, inflation="logit")
            else:
                zi = ZeroInflatedNegativeBinomialP(y, exog, exog_infl=exog_infl, inflation="logit", p=2)
            res = zi.fit(method="bfgs", maxiter=5000, gtol=1e-10, disp=False)
            q = exog_infl.shape[1]
            params = np.asarray(res.params)
            alpha = params[q + exog.shape[1]] if family == "negative-binomial" else 0.0
            out.append(
                case(
                    f"zip/{family}/{name}",
                    "zeroInflated",
                    [x, y],
                    {
                        "coef": params[q : q + d],
                        "intercept": np.array(float(params[q + d])),
                        "zeroCoef": params[: q - 1],
                        "zeroIntercept": np.array(float(params[q - 1])),
                        "alpha": np.array(float(alpha)),
                        "logLikelihood": np.array(float(res.llf)),
                    },
                    rtol=1e-4,
                    atol=1e-5,
                    family=family,
                    inflation=1.0 if infl else 0.0,
                )
            )
        # Hurdle: the zero part and the truncated count part are fitted separately.
        x, y = draw(g, 500, 2, family)
        exog = np.column_stack([x, np.ones(len(y))])
        exog_zero = np.column_stack([x[:, [0]], np.ones(len(y))])
        zero_fit = sm.Logit((y == 0).astype(np.float64), exog_zero).fit(disp=False, gtol=1e-12, maxiter=500)
        pos = y > 0
        if family == "poisson":
            cnt = TruncatedLFPoisson(y[pos], exog[pos], truncation=0).fit(
                method="bfgs", maxiter=5000, gtol=1e-10, disp=False
            )
            alpha = 0.0
        else:
            cnt = TruncatedLFNegativeBinomialP(y[pos], exog[pos], truncation=0, p=2).fit(
                method="bfgs", maxiter=5000, gtol=1e-10, disp=False
            )
            alpha = float(np.asarray(cnt.params)[-1])
        cp = np.asarray(cnt.params)
        zp = np.asarray(zero_fit.params)
        out.append(
            case(
                f"hurdle/{family}",
                "hurdle",
                [x, y],
                {
                    "coef": cp[:2],
                    "intercept": np.array(float(cp[2])),
                    "zeroCoef": zp[:1],
                    "zeroIntercept": np.array(float(zp[1])),
                    "alpha": np.array(alpha),
                    "logLikelihood": np.array(float(zero_fit.llf) + float(cnt.llf)),
                },
                rtol=1e-4,
                atol=1e-5,
                family=family,
            )
        )
    return out
