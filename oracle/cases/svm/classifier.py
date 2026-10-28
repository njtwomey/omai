"""SVC and NuSVC against scikit-learn (libsvm): support vectors, dual coefficients, intercepts and decision values."""

from typing import cast

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.svm import SVC, NuSVC

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# libsvm keeps its kernel cache in single precision, so agreement is to about 1e-5.
RTOL = 1e-4
ATOL = 1e-5


def blobs(
    g: np.random.Generator, centres: list[tuple[float, float]], per: int, spread: float
) -> tuple[FloatArray, FloatArray]:
    x = np.vstack([g.normal(c, spread, size=(per, 2)) for c in centres])
    y = np.repeat(np.arange(len(centres), dtype=np.float64), per)
    return x, y


def record(
    name: str,
    model: SVC | NuSVC,
    xt: FloatArray,
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray | None = None,
    **params: object,
) -> Case:
    output = {
        "support": np.asarray(model.support_, dtype=np.float64),
        "nSupport": np.asarray(model.n_support_, dtype=np.float64),
        "dualCoef": np.asarray(model.dual_coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
        "ovo": np.asarray(model.decision_function(xt) if len(model.classes_) == 2 else decision_ovo(model, xt)),
        "predict": np.asarray(model.predict(xt), dtype=np.float64),
    }
    if len(model.classes_) > 2:
        output["ovr"] = np.asarray(model.decision_function(xt), dtype=np.float64)
    inputs = [x, y, xt] if weights is None else [x, y, xt, weights]
    # The nu form divides by the margin, which amplifies the single-precision rounding.
    loose = isinstance(model, NuSVC)
    return case(
        f"svc/{name}", "svc", inputs, output, rtol=1e-3 if loose else RTOL, atol=5e-4 if loose else ATOL, **params
    )


def decision_ovo(model: SVC | NuSVC, xt: FloatArray) -> FloatArray:
    shape = model.decision_function_shape
    model.decision_function_shape = "ovo"
    out = np.asarray(model.decision_function(xt), dtype=np.float64)
    model.decision_function_shape = shape
    return out


def scalar_gamma(value: float) -> str:
    """scikit-learn takes a number for gamma; its type stubs only declare the string forms."""
    return cast("str", value)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(31)
    x2, y2 = blobs(g, [(-0.8, 0.0), (0.8, 0.4)], 14, 1.0)
    t2 = g.uniform(-3, 3, size=(25, 2))
    for c in (0.5, 10.0):
        rbf = SVC(C=c, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8).fit(x2, y2)
        out.append(record(f"binary-rbf-C{c}", rbf, t2, x2, y2, form="c", C=c, kernel="rbf", gamma=0.5))
        lin = SVC(C=c, kernel="linear", tol=1e-8).fit(x2, y2)
        out.append(record(f"binary-linear-C{c}", lin, t2, x2, y2, form="c", C=c, kernel="linear"))
        poly = SVC(C=c, kernel="poly", gamma=scalar_gamma(0.4), degree=3, coef0=1.0, tol=1e-8).fit(x2, y2)
        out.append(
            record(f"binary-poly-C{c}", poly, t2, x2, y2, form="c", C=c, kernel="poly", gamma=0.4, degree=3, coef0=1.0)
        )
    model = SVC(C=2.0, kernel="rbf", gamma="scale", tol=1e-8).fit(x2, y2)
    out.append(record("binary-scale", model, t2, x2, y2, form="c", C=2.0, kernel="rbf", gamma="scale"))
    model = SVC(C=2.0, kernel="rbf", gamma="auto", tol=1e-8).fit(x2, y2)
    out.append(record("binary-auto", model, t2, x2, y2, form="c", C=2.0, kernel="rbf", gamma="auto"))
    x3, y3 = blobs(g, [(-1.5, 0.0), (1.5, 0.0), (0.0, 2.0)], 12, 0.9)
    t3 = g.uniform(-3, 3, size=(30, 2))
    model = SVC(C=1.0, kernel="rbf", gamma=scalar_gamma(0.6), tol=1e-8).fit(x3, y3)
    out.append(record("three-classes", model, t3, x3, y3, form="c", C=1.0, kernel="rbf", gamma=0.6))
    x4, y4 = blobs(g, [(-1.5, 0.0), (1.5, 0.0), (0.0, 2.0), (0.0, -2.0)], 8, 0.9)
    t4 = g.uniform(-3, 3, size=(30, 2))
    model = SVC(C=3.0, kernel="rbf", gamma=scalar_gamma(0.3), tol=1e-8).fit(x4, y4)
    out.append(record("four-classes", model, t4, x4, y4, form="c", C=3.0, kernel="rbf", gamma=0.3))
    # Labels that are not 0..k-1, and class weights.
    y_odd = np.where(y2 == 0, -3.0, 7.0)
    model = SVC(C=1.0, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8, class_weight={-3.0: 1.0, 7.0: 4.0}).fit(
        x2, y_odd
    )
    out.append(
        record(
            "class-weight",
            model,
            t2,
            x2,
            y_odd,
            form="c",
            C=1.0,
            kernel="rbf",
            gamma=0.5,
            classWeight={"-3": 1.0, "7": 4.0},
        )
    )
    model = SVC(C=1.0, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8, class_weight="balanced").fit(x3, y3)
    out.append(record("balanced", model, t3, x3, y3, form="c", C=1.0, kernel="rbf", gamma=0.5, classWeight="balanced"))
    weights = g.uniform(0.2, 3.0, size=x2.shape[0])
    model = SVC(C=1.0, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8).fit(x2, y2, sample_weight=weights)
    out.append(record("sample-weight", model, t2, x2, y2, weights, form="c", C=1.0, kernel="rbf", gamma=0.5))
    for nu in (0.2, 0.5):
        nu_model = NuSVC(nu=nu, kernel="rbf", gamma=scalar_gamma(0.5), tol=1e-8).fit(x2, y2)
        out.append(record(f"nu-binary-{nu}", nu_model, t2, x2, y2, form="nu", nu=nu, kernel="rbf", gamma=0.5))
    nu_model = NuSVC(nu=0.3, kernel="rbf", gamma=scalar_gamma(0.6), tol=1e-8).fit(x3, y3)
    out.append(record("nu-three-classes", nu_model, t3, x3, y3, form="nu", nu=0.3, kernel="rbf", gamma=0.6))
    return out
