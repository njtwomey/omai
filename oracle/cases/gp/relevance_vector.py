"""The relevance vector machine against scikit-learn ARDRegression on the kernel matrix (the same evidence updates)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.linear_model import ARDRegression
from sklearn.metrics.pairwise import rbf_kernel

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(83)
    x = np.sort(g.uniform(-3.0, 3.0, size=(30, 1)), axis=0)
    y = np.sinc(x[:, 0]) + 0.05 * g.normal(size=30)
    xt = np.linspace(-3.0, 3.0, 11).reshape(-1, 1)
    for name, gamma, intercept in [("rbf", 0.5, True), ("narrow", 2.0, True), ("no-intercept", 0.5, False)]:
        gram: FloatArray = np.asarray(rbf_kernel(x, x, gamma=gamma), dtype=np.float64)
        cross: FloatArray = np.asarray(rbf_kernel(xt, x, gamma=gamma), dtype=np.float64)
        model = ARDRegression(max_iter=3000, tol=1e-10, fit_intercept=intercept).fit(gram, y)
        keep = model.lambda_ < model.threshold_lambda
        output = {
            "relevant": np.flatnonzero(keep).astype(np.float64),
            "coef": np.asarray(model.coef_, dtype=np.float64),
            "intercept": np.array(model.intercept_),
            "noise": np.array(model.alpha_),
            "mean": np.asarray(model.predict(cross), dtype=np.float64),
        }
        out.append(
            case(
                f"rvm/{name}",
                "relevanceVectorMachine",
                [x, y, xt],
                output,
                rtol=1e-5,
                atol=1e-6,
                lengthScale=float(1 / np.sqrt(2 * gamma)),
                fitIntercept=intercept,
            )
        )
    return out
