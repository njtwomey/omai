"""Support vector machines against scikit-learn (libsvm and LIBLINEAR, tight tolerances): SVC decision values and
support sizes for two and three classes, SVR predictions, LinearSVC coefficients and OneClassSVM decision values."""

import numpy as np
import sklearn
from sklearn.svm import SVC, SVR, LinearSVC, OneClassSVM
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(91)
    x = g.normal(size=(50, 2))
    y2 = (x[:, 0] ** 2 + x[:, 1] > 0.5).astype(np.float64)
    y3 = np.digitize(x[:, 0] + 0.6 * x[:, 1], [-0.5, 0.5]).astype(np.float64)
    t = np.sin(x[:, 0]) + 0.3 * x[:, 1]
    q = g.normal(size=(6, 2))
    out: list[Case] = []
    for kernel in ["rbf", "linear", "poly"]:
        s = SVC(C=2.0, kernel=kernel, gamma="scale", tol=1e-8).fit(x, y2)
        out.append(
            case(
                f"svc/{kernel}",
                "svc",
                [x, y2, q],
                {"decision": s.decision_function(q), "nSupport": s.n_support_.astype(np.float64)},
                rtol=1e-5,
                atol=1e-6,
                kernel=kernel,
            )
        )
    s3 = SVC(C=1.0, tol=1e-8).fit(x, y3)
    out.append(case("svc/multiclass", "svcMulti", [x, y3, q], {"predict": s3.predict(q)}))
    r = SVR(C=3.0, epsilon=0.05, tol=1e-8).fit(x, t)
    out.append(case("svr", "svr", [x, t, q], {"predict": r.predict(q)}, rtol=1e-5, atol=1e-6))
    lin = LinearSVC(C=0.7, tol=1e-12, max_iter=1000000).fit(x, y2)
    out.append(
        case(
            "linearSvc", "linearSvc", [x, y2], {"coef": lin.coef_[0], "intercept": lin.intercept_}, rtol=1e-6, atol=1e-7
        )
    )
    oc = OneClassSVM(nu=0.3, gamma="scale", tol=1e-8).fit(x)
    out.append(case("oneClass", "oneClass", [x, q], {"decision": oc.decision_function(q)}, rtol=1e-5, atol=1e-6))
    return out
