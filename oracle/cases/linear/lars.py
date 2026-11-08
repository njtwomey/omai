"""Least angle regression against sklearn.linear_model.lars_path: the knots (penalties and coefficients) and the order
in which variables enter, for the plain LARS and the lasso modification, and LassoLars at a fixed penalty."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.linear_model import LassoLars, lars_path

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(2004)
    out: list[Case] = []
    for name, n, d in [("small", 30, 4), ("wide", 40, 8)]:
        x = g.normal(size=(n, d))
        x[:, 1] += 0.8 * x[:, 0]  # correlated columns make the lasso path drop variables
        x[:, 3] -= 0.6 * x[:, 2]
        beta = np.zeros(d)
        beta[:3] = [3.0, -2.0, 1.5]
        y = x @ beta + 0.5 * g.normal(size=n)
        xc = x - x.mean(axis=0)
        yc = y - y.mean()
        for method in ["lar", "lasso"]:
            alphas, active, coefs = lars_path(xc, yc, method=method)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"path/{name}/{method}",
                    "path",
                    [x, y],
                    {"alphas": alphas, "coefs": coefs.T, "order": np.array(active, dtype=np.float64)},
                    rtol=1e-8,
                    atol=1e-9,
                    method=method,
                )
            )
        for alpha in (0.4, 0.05):
            m = LassoLars(alpha=alpha, fit_intercept=True).fit(x, y)
            out.append(
                case(
                    f"lassolars/{name}/{alpha}",
                    "lassoLars",
                    [x, y],
                    {"coef": m.coef_, "intercept": np.array(float(m.intercept_))},  # pyright: ignore[reportArgumentType]
                    rtol=1e-8,
                    atol=1e-9,
                    alpha=alpha,
                )
            )
    return out
