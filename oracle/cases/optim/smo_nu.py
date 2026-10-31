"""The nu variants and the one-class dual of SMO against libsvm through scikit-learn, on precomputed RBF kernels."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.svm import NuSVC, NuSVR, OneClassSVM

# libsvm keeps its kernel cache in single precision, and nu-SVC divides by the margin,
# so the nu forms agree to 1e-3 and 1e-5.
REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def rbf(x: FloatArray, gamma: float) -> FloatArray:
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-gamma * d)


def expansion(n: int, support: object, dual: object) -> FloatArray:
    coef = np.zeros(n)
    coef[np.asarray(support, dtype=np.int64)] = np.asarray(dual).ravel()
    return coef


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(23)
    overlapping = np.vstack([g.normal(-0.6, 1.0, size=(14, 2)), g.normal(0.6, 1.0, size=(14, 2))])
    n = overlapping.shape[0]
    y = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    k = rbf(overlapping, 0.5)
    for nu in (0.2, 0.5, 0.8):
        clf = NuSVC(nu=nu, kernel="precomputed", tol=1e-10)
        clf.fit(k, y)
        output = {
            "coef": expansion(n, clf.support_, clf.dual_coef_),
            "intercept": np.array(clf.intercept_[0]),
            "decision": clf.decision_function(k),
        }
        out.append(case(f"nu-svc/{nu}", "nuSvcDual", [k, y], output, rtol=1e-3, atol=1e-3, nu=nu))
    xr = np.sort(g.uniform(-3.0, 3.0, size=(18, 1)), axis=0)
    t = np.sin(xr[:, 0]) + 0.1 * g.normal(size=18)
    kr = rbf(xr, 1.0)
    for nu in (0.3, 0.7):
        reg = NuSVR(nu=nu, C=2.0, kernel="precomputed", tol=1e-10)
        reg.fit(kr, t)
        output = {
            "coef": expansion(18, reg.support_, reg.dual_coef_),
            "intercept": np.array(reg.intercept_[0]),
            "decision": reg.predict(kr),
        }
        out.append(case(f"nu-svr/{nu}", "nuSvrDual", [kr, t], output, rtol=1e-5, atol=1e-5, nu=nu, C=2.0))
    x1 = np.vstack([g.normal(0.0, 1.0, size=(30, 2)), g.normal(3.0, 0.3, size=(4, 2))])
    k1 = rbf(x1, 0.4)
    for nu in (0.1, 0.3):
        one = OneClassSVM(nu=nu, kernel="precomputed", tol=1e-10)
        one.fit(k1)
        output = {
            "coef": expansion(x1.shape[0], one.support_, one.dual_coef_),
            "intercept": np.array(one.intercept_[0]),
            "decision": one.decision_function(k1),
        }
        out.append(case(f"one-class/{nu}", "oneClassDual", [k1], output, rtol=1e-6, atol=1e-6, nu=nu))
    return out
