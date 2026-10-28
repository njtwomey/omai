"""OneClassSVM against scikit-learn (libsvm): support vectors, dual coefficients, offset and decision values."""

from typing import cast

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.svm import OneClassSVM

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# libsvm keeps its kernel cache in single precision.
RTOL = 1e-4
ATOL = 1e-5


def scalar_gamma(value: float) -> str:
    """scikit-learn takes a number for gamma; its type stubs only declare the string forms."""
    return cast("str", value)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(41)
    x = np.vstack([g.normal(0.0, 1.0, size=(40, 2)), g.normal(3.0, 0.3, size=(5, 2))])
    xt = g.uniform(-4.0, 5.0, size=(30, 2))
    for nu in (0.1, 0.3, 0.6):
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=scalar_gamma(0.4), tol=1e-8).fit(x)
        output = {
            "support": np.asarray(model.support_, dtype=np.float64),
            "dualCoef": np.asarray(model.dual_coef_, dtype=np.float64),
            "intercept": np.asarray(model.intercept_, dtype=np.float64),
            "decision": np.asarray(model.decision_function(xt), dtype=np.float64),
            "predict": np.asarray(model.predict(xt), dtype=np.float64),
        }
        out.append(
            case(f"one-class/rbf-nu{nu}", "oneClassSvm", [x, xt], output, rtol=RTOL, atol=ATOL, nu=nu, gamma=0.4)
        )
    model = OneClassSVM(nu=0.2, kernel="poly", degree=2, gamma=scalar_gamma(0.3), coef0=1.0, tol=1e-8).fit(x)
    output = {
        "support": np.asarray(model.support_, dtype=np.float64),
        "dualCoef": np.asarray(model.dual_coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
        "decision": np.asarray(model.decision_function(xt), dtype=np.float64),
        "predict": np.asarray(model.predict(xt), dtype=np.float64),
    }
    out.append(
        case(
            "one-class/poly",
            "oneClassSvm",
            [x, xt],
            output,
            rtol=1e-3,
            atol=1e-4,
            nu=0.2,
            kernel="poly",
            gamma=0.3,
            degree=2,
            coef0=1.0,
        )
    )
    return out
