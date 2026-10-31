"""Robust linear regression against scikit-learn: TheilSenRegressor (every subset, so the spatial median iteration is
the only approximation and is matched step for step), HuberRegressor and QuantileRegressor (HiGHS), on data with
outliers in the targets."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.linear_model import HuberRegressor, QuantileRegressor, TheilSenRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def data(g: np.random.Generator, n: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    x = g.normal(size=(n, d))
    beta = np.linspace(2.0, -1.0, d)
    y = x @ beta + 1.5 + 0.3 * g.normal(size=n)
    y[:3] += np.array([15.0, -12.0, 20.0])  # outliers
    return x, y


def cases() -> list[Case]:
    g = rng(1968)
    out: list[Case] = []
    for name, n, d in [("one", 24, 1), ("two", 20, 2)]:
        x, y = data(g, n, d)
        m = TheilSenRegressor(random_state=0, max_subpopulation=100000).fit(x, y)
        out.append(
            case(
                f"theilsen/{name}",
                "theilSen",
                [x, y],
                {"coef": m.coef_, "intercept": np.array(float(m.intercept_))},  # pyright: ignore[reportArgumentType]
                rtol=1e-8,
                atol=1e-9,
            )
        )
    for name, n, d in [("small", 30, 2), ("wide", 60, 4)]:
        x, y = data(g, n, d)
        for alpha, eps in [(1e-4, 1.35), (0.0, 1.35), (0.5, 1.5)]:
            m = HuberRegressor(epsilon=eps, alpha=alpha, tol=1e-14, max_iter=100000).fit(x, y)
            out.append(
                case(
                    f"huber/{name}/{alpha}/{eps}",
                    "huber",
                    [x, y],
                    {
                        "coef": m.coef_,
                        "intercept": np.array(float(m.intercept_)),  # pyright: ignore[reportArgumentType]
                        "scale": np.array(float(m.scale_)),
                    },
                    rtol=1e-5,
                    atol=1e-6,
                    alpha=alpha,
                    epsilon=eps,
                )
            )
        for tau, alpha in [(0.5, 0.0), (0.2, 0.0), (0.8, 0.05)]:
            m = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs").fit(x, y)
            out.append(
                case(
                    f"quantile/{name}/{tau}/{alpha}",
                    "quantile",
                    [x, y],
                    {"coef": m.coef_, "intercept": np.array(float(m.intercept_))},  # pyright: ignore[reportArgumentType]
                    rtol=1e-6,
                    atol=1e-7,
                    quantile=tau,
                    alpha=alpha,
                )
            )
    return out
