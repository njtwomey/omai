"""Variance-reduced finite-sum methods (optim/variance-reduced.ts) against scikit-learn's SAG and SAGA solvers.

The converged coefficients of ridge regression and L2- and L1-penalised logistic regression (no intercept) are the
references; every method that converges to the unique minimiser should match them.
"""

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(5)
    n, d = 40, 3
    x = g.normal(size=(n, d))
    w = np.array([1.5, -2.0, 0.5])
    y_reg = x @ w + 0.3 * g.normal(size=n)
    y_cls = np.where(x @ w + 0.5 * g.normal(size=n) > 0, 1.0, -1.0)

    ridge = Ridge(alpha=2.0, solver="sag", fit_intercept=False, tol=1e-12, max_iter=200000, random_state=0)
    ridge.fit(x, y_reg)
    l2 = LogisticRegression(C=0.5, solver="sag", fit_intercept=False, tol=1e-12, max_iter=200000, random_state=0)
    l2.fit(x, y_cls)
    l1 = LogisticRegression(
        C=0.3, l1_ratio=1.0, solver="saga", fit_intercept=False, tol=1e-12, max_iter=200000, random_state=0
    )
    l1.fit(x, y_cls)
    return [
        case("ridge", "ridge", [x, y_reg], np.asarray(ridge.coef_), rtol=1e-6, atol=1e-7, alpha=2.0),
        case("logistic/l2", "logisticL2", [x, y_cls], np.asarray(l2.coef_).ravel(), rtol=1e-6, atol=1e-7, C=0.5),
        case("logistic/l1", "logisticL1", [x, y_cls], np.asarray(l1.coef_).ravel(), rtol=1e-6, atol=1e-7, C=0.3),
    ]
