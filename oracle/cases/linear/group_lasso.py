"""The multi-task lasso and elastic net against scikit-learn's MultiTaskLasso and MultiTaskElasticNet: coefficients
(a row per task) and intercepts at several penalties, for correlated features and a shared support."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(2006)
    out: list[Case] = []
    for name, n, d, k in [("small", 30, 5, 3), ("wide", 40, 12, 4)]:
        x = g.normal(size=(n, d))
        x[:, 1] += 0.8 * x[:, 0]
        b = np.zeros((d, k))
        b[:3, :] = g.normal(size=(3, k)) * 2.0
        y = x @ b + 3.0 + 0.5 * g.normal(size=(n, k))
        for alpha, ratio in [(0.3, 1.0), (0.05, 1.0), (0.2, 0.6)]:
            if ratio == 1.0:
                m = MultiTaskLasso(alpha=alpha, tol=1e-14, max_iter=100000).fit(x, y)
            else:
                m = MultiTaskElasticNet(alpha=alpha, l1_ratio=ratio, tol=1e-14, max_iter=100000).fit(x, y)
            out.append(
                case(
                    f"multitask/{name}/{alpha}/{ratio}",
                    "multiTask",
                    [x, y],
                    {"coef": m.coef_, "intercept": m.intercept_},  # pyright: ignore[reportArgumentType]
                    rtol=1e-6,
                    atol=1e-7,
                    alpha=alpha,
                    l1Ratio=ratio,
                )
            )
    return out
