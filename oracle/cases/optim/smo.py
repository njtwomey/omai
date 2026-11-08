"""SMO against libsvm through scikit-learn (SVC and SVR on precomputed RBF kernels): coefficients and intercepts."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.svm import SVC, SVR

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def rbf(x: FloatArray, gamma: float) -> FloatArray:
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-gamma * d)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(17)
    separable = np.vstack([g.normal(-2.0, 0.6, size=(10, 2)), g.normal(2.0, 0.6, size=(10, 2))])
    overlapping = np.vstack([g.normal(-0.5, 1.0, size=(14, 2)), g.normal(0.5, 1.0, size=(14, 2))])
    for name, x, c, weight in [
        ("separable", separable, 10.0, 1.0),
        ("overlapping", overlapping, 1.0, 1.0),
        ("weighted", overlapping, 0.5, 3.0),
    ]:
        n = x.shape[0]
        y = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        k = rbf(x, 0.5)
        clf = SVC(C=c, kernel="precomputed", tol=1e-10, class_weight={1.0: weight})
        clf.fit(k, y)
        coef = np.zeros(n)
        coef[clf.support_] = np.asarray(clf.dual_coef_).ravel()
        cs = np.where(y > 0, c * weight, c)
        output = {"coef": coef, "intercept": np.array(clf.intercept_[0]), "decision": clf.decision_function(k)}
        out.append(case(f"svc/{name}", "svcDual", [k, y, cs], output, rtol=1e-6, atol=1e-6))
    xr = np.sort(g.uniform(-3.0, 3.0, size=(16, 1)), axis=0)
    t = np.sin(xr[:, 0]) + 0.1 * g.normal(size=16)
    k = rbf(xr, 1.0)
    reg = SVR(C=2.0, epsilon=0.1, kernel="precomputed", tol=1e-10)
    reg.fit(k, t)
    coef = np.zeros(16)
    coef[reg.support_] = np.asarray(reg.dual_coef_).ravel()
    output = {"coef": coef, "intercept": np.array(reg.intercept_[0]), "decision": reg.predict(k)}
    out.append(case("svr/sine", "svrDual", [k, t], output, rtol=1e-6, atol=1e-6, C=2.0, epsilon=0.1))
    return out
