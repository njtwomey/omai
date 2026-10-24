"""OWL-QN against scikit-learn's L1-regularised solvers: the lasso by coordinate descent and L1 logistic regression by
liblinear, on seeded data with sparse ground truth (the solutions are unique, so they are compared directly)."""

import numpy as np
import sklearn
from sklearn.linear_model import Lasso, LogisticRegression
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(7)
    n, p = 40, 8
    x = g.normal(size=(n, p))
    w = np.array([1.5, 0.0, -2.0, 0.0, 0.0, 0.7, 0.0, 0.0])
    y = x @ w + 0.3 * g.normal(size=n)
    for alpha in [0.05, 0.2, 0.8]:
        model = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, max_iter=1_000_000).fit(x, y)
        out.append(case(f"lasso/{alpha}", "lasso", [x, y], np.asarray(model.coef_), rtol=1e-6, atol=1e-7, alpha=alpha))
    labels = np.where(x @ w + 0.5 * g.normal(size=n) > 0, 1.0, -1.0)
    for c in [0.1, 1.0]:
        model = LogisticRegression(
            penalty="l1", C=c, fit_intercept=False, solver="liblinear", tol=1e-12, max_iter=100_000, random_state=0
        ).fit(x, labels)
        out.append(
            case(f"logistic/{c}", "logistic", [x, labels], np.asarray(model.coef_[0]), rtol=1e-5, atol=1e-6, C=c)
        )
    return out
