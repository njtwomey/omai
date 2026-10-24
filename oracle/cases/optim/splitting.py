"""Proximal gradient and primal-dual solvers on the lasso and the elastic net against sklearn's coordinate descent."""

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet, Lasso
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}


def lasso_cases() -> list[Case]:
    """min |A x - b|^2 / 2 + lam |x|_1: sklearn's Lasso with alpha = lam / m and no intercept."""
    g = rng(43)
    out: list[Case] = []
    m, n = 30, 8
    a = g.normal(size=(m, n))
    truth = np.array([2.0, 0.0, -1.5, 0.0, 0.0, 0.7, 0.0, 0.0])
    b = a @ truth + g.normal(size=m) * 0.1
    for i, lam in enumerate([0.5, 3.0]):
        model = Lasso(alpha=lam / m, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
        model.fit(a, b)
        coef = np.asarray(model.coef_, dtype=np.float64)
        out.append(case(f"lasso/{i}", "lasso", [a, b], coef, rtol=1e-6, atol=1e-7, lam=lam))
    net = ElasticNet(alpha=1.0 / m, l1_ratio=0.6, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
    net.fit(a, b)
    # sklearn: |b - A x|^2 / (2 m) + alpha l1_ratio |x|_1 + alpha (1 - l1_ratio) |x|^2 / 2; scaled by m.
    out.append(
        case(
            "elasticNet/0",
            "elasticNet",
            [a, b],
            np.asarray(net.coef_, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
            l1=0.6,
            l2=0.4,
        )
    )
    return out


def cases() -> list[Case]:
    return lasso_cases()
