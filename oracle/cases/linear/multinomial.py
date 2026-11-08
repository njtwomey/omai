"""The multinomial logit against statsmodels' MNLogit (unpenalised, class 0 as the reference: coefficients, standard
errors, log-likelihood and probabilities) and scikit-learn's multinomial LogisticRegression (ridge penalty). With a
ridge penalty the intercepts are only determined up to a common shift, so they are compared centred."""

import numpy as np
import sklearn
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import MNLogit

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__}


def draw(g: np.random.Generator, n: int, d: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = g.normal(size=(n, d))
    w = g.normal(size=(d, k))
    logits = x @ w + g.gumbel(size=(n, k))
    return x, np.argmax(logits, axis=1).astype(np.float64)


def cases() -> list[Case]:
    g = rng(1973)
    out: list[Case] = []
    for name, n, d, k in [("three", 200, 2, 3), ("four", 300, 3, 4)]:
        x, y = draw(g, n, d, k)
        exog = sm.add_constant(x, prepend=False)
        res = MNLogit(y, exog).fit(method="newton", disp=False, maxiter=200, tol=1e-12)
        params = np.asarray(res.params)  # [d + 1, k - 1]
        coef = np.vstack([np.zeros(d), params[:d].T])
        intercept = np.concatenate([[0.0], params[d]])
        out.append(
            case(
                f"mnlogit/{name}",
                "multinomial",
                [x, y],
                {
                    "coef": coef,
                    "intercept": intercept,
                    "logLikelihood": np.array(float(res.llf)),
                    "bse": np.asarray(res.bse).T.reshape(-1),
                    "proba": np.asarray(res.predict(exog[:8])),
                },
                rtol=1e-7,
                atol=1e-8,
                alpha=0.0,
            )
        )
        for c in (0.1, 5.0):
            m = LogisticRegression(C=c, tol=1e-12, max_iter=10000).fit(x, y)
            out.append(
                case(
                    f"ridge/{name}/{c}",
                    "multinomial",
                    [x, y],
                    {
                        "coef": m.coef_,
                        "intercept": m.intercept_ - m.intercept_.mean(),
                        "proba": m.predict_proba(x[:8]),
                    },
                    rtol=1e-6,
                    atol=1e-7,
                    alpha=1.0 / (c * n),
                )
            )
    return out
