"""Least angle regression against scikit-learn's lars_path (LAR and the lasso modification) and LassoLars."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.linear_model import LassoLars, lars_path

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(131)
    n = 40
    z = g.normal(size=(n, 5))
    x = z + 0.6 * z[:, [0]]
    y = x @ np.array([2.0, 0.0, -1.5, 0.5, 0.0]) + 0.5 * g.normal(size=n)
    xc = x - x.mean(0)
    yc = y - y.mean()
    out: list[Case] = []
    for method in ["lar", "lasso"]:
        alphas, _, coefs = lars_path(xc, yc, method=method)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"path/{method}",
                "path",
                [x, y],
                {"alphas": alphas, "coefs": coefs.T},
                rtol=1e-9,
                atol=1e-10,
                method=method,
            )
        )
    ll = LassoLars(alpha=0.05).fit(x, y)
    out.append(
        case(
            "lassoLars",
            "lassoLars",
            [x, y],
            {"coef": ll.coef_, "intercept": np.array(ll.intercept_)},
            rtol=1e-9,
            atol=1e-10,
        )
    )
    return out
