"""SVR and NuSVR against scikit-learn (libsvm): support vectors, dual coefficients, intercepts and predictions."""

from typing import cast

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.svm import SVR, NuSVR

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# libsvm keeps its kernel cache in single precision, so agreement is to about 1e-5.
RTOL = 1e-3
ATOL = 5e-5


def record(
    name: str,
    model: SVR | NuSVR,
    x: FloatArray,
    y: FloatArray,
    xt: FloatArray,
    weights: FloatArray | None = None,
    **params: object,
) -> Case:
    output = {
        "support": np.asarray(model.support_, dtype=np.float64),
        "dualCoef": np.asarray(model.dual_coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
        "predict": np.asarray(model.predict(xt), dtype=np.float64),
    }
    inputs = [x, y, xt] if weights is None else [x, y, xt, weights]
    loose = isinstance(model, NuSVR)
    return case(
        f"svr/{name}", "svr", inputs, output, rtol=1e-3 if loose else RTOL, atol=5e-4 if loose else ATOL, **params
    )


def scalar_gamma(value: float) -> str:
    """scikit-learn takes a number for gamma; its type stubs only declare the string forms."""
    return cast("str", value)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(37)
    x = np.sort(g.uniform(-3.0, 3.0, size=(40, 1)), axis=0)
    y = np.sin(x[:, 0]) + 0.15 * g.normal(size=40)
    xt = np.linspace(-3.5, 3.5, 25).reshape(-1, 1)
    for eps in (0.05, 0.3):
        rbf = SVR(C=2.0, epsilon=eps, kernel="rbf", gamma=scalar_gamma(1.0), tol=1e-8).fit(x, y)
        out.append(record(f"rbf-eps{eps}", rbf, x, y, xt, form="epsilon", C=2.0, epsilon=eps, kernel="rbf", gamma=1.0))
    lin = SVR(C=5.0, epsilon=0.2, kernel="linear", tol=1e-8).fit(x, y)
    out.append(record("linear", lin, x, y, xt, form="epsilon", C=5.0, epsilon=0.2, kernel="linear"))
    poly = SVR(C=5.0, epsilon=0.1, kernel="poly", gamma=scalar_gamma(0.5), degree=3, coef0=1.0, tol=1e-8).fit(x, y)
    out.append(
        record(
            "poly", poly, x, y, xt, form="epsilon", C=5.0, epsilon=0.1, kernel="poly", gamma=0.5, degree=3, coef0=1.0
        )
    )
    scale = SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="scale", tol=1e-8).fit(x, y)
    out.append(record("scale", scale, x, y, xt, form="epsilon", C=1.0, epsilon=0.1, kernel="rbf", gamma="scale"))
    weights = g.uniform(0.2, 3.0, size=40)
    weighted = SVR(C=2.0, epsilon=0.1, kernel="rbf", gamma=scalar_gamma(1.0), tol=1e-8).fit(x, y, sample_weight=weights)
    out.append(
        record(
            "sample-weight", weighted, x, y, xt, weights, form="epsilon", C=2.0, epsilon=0.1, kernel="rbf", gamma=1.0
        )
    )
    x2 = g.uniform(-2.0, 2.0, size=(50, 2))
    y2 = x2[:, 0] ** 2 - x2[:, 1] + 0.1 * g.normal(size=50)
    xt2 = g.uniform(-2.0, 2.0, size=(20, 2))
    wide = SVR(C=3.0, epsilon=0.1, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8).fit(x2, y2)
    out.append(record("two-features", wide, x2, y2, xt2, form="epsilon", C=3.0, epsilon=0.1, kernel="rbf", gamma=0.5))
    for nu in (0.2, 0.6):
        nu_model = NuSVR(nu=nu, C=2.0, kernel="rbf", gamma=scalar_gamma(1.0), tol=1e-8).fit(x, y)
        out.append(record(f"nu-{nu}", nu_model, x, y, xt, form="nu", nu=nu, C=2.0, kernel="rbf", gamma=1.0))
    return out
