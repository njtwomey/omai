"""Robust regression against scikit-learn: Huber, Theil-Sen (every subset) and quantile regression."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.linear_model import HuberRegressor, QuantileRegressor, TheilSenRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(21)
    n = 40
    x = g.normal(size=(n, 2))
    y = 1.0 + x @ np.array([2.0, -1.0]) + 0.3 * g.normal(size=n)
    y[:4] += np.array([8.0, -6.0, 10.0, 7.0])
    out: list[Case] = []
    h = HuberRegressor(epsilon=1.35, alpha=1e-4, tol=1e-10, max_iter=10000).fit(x, y)
    out.append(
        case(
            "huber",
            "huber",
            [x, y],
            {"coef": h.coef_, "intercept": np.array(h.intercept_), "scale": np.array(h.scale_)},
            rtol=1e-5,
            atol=1e-6,
        )
    )
    x1 = x[:20, :1]
    ts = TheilSenRegressor(random_state=0).fit(x1, y[:20])
    out.append(
        case(
            "theilSen",
            "theilSen",
            [x1, y[:20]],
            {"coef": ts.coef_, "intercept": np.array(ts.intercept_)},
            rtol=1e-9,
            atol=1e-9,
        )
    )
    for q in [0.5, 0.8]:
        qr = QuantileRegressor(quantile=q, alpha=0.01, solver="highs").fit(x, y)
        out.append(
            case(
                f"quantile/{q}",
                "quantile",
                [x, y],
                {"coef": qr.coef_, "intercept": np.array(qr.intercept_)},
                rtol=1e-7,
                atol=1e-8,
                quantile=q,
                alpha=0.01,
            )
        )
    return out
