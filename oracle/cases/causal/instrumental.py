"""Instrumental-variable estimators: two-stage least squares against statsmodels' IV2SLS (coefficients and classical
errors), the first-stage F statistic against statsmodels' OLS F test, and LIML and Fuller's estimator from the k-class
formulas with the variance-ratio root taken by scipy's generalised symmetric eigensolver (neither statsmodels nor
linearmodels is available for them)."""

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.linalg import eigh
from scipy.stats import chi2
from statsmodels.sandbox.regression.gmm import IV2SLS

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__, "scipy": scipy.__version__}


def design(seed: int, n: int, m: int, c: int, strength: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Outcome, endogenous regressor, instruments `[n, m]` and controls `[n, c]` with an unobserved confounder."""
    g = rng(seed)
    z = g.normal(size=(n, m))
    w = g.normal(size=(n, c))
    u = g.normal(size=n)
    d = z @ np.linspace(1.0, 0.4, m) * strength + 0.5 * w.sum(axis=1) + u + g.normal(size=n)
    y = 1.5 * d + w @ np.linspace(0.5, -0.3, c) + 2.0 * u + g.normal(size=n)
    return y, d, z, w


def project(a: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(a - w @ np.linalg.lstsq(w, a, rcond=None)[0], dtype=np.float64)


def kclass(
    y: FloatArray, d: FloatArray, z: FloatArray, w: FloatArray, kappa: float | None, fuller: float | None
) -> tuple[FloatArray, FloatArray, float]:
    n = len(y)
    x = np.column_stack([d, w, np.ones(n)])
    included = np.column_stack([w, np.ones(n)])
    full = np.column_stack([included, z])
    root = 1.0
    if kappa is None:
        yy = np.column_stack([y, d])
        a = project(yy, full).T @ project(yy, full)
        b = project(yy, included).T @ project(yy, included)
        root = float(eigh(b, a, eigvals_only=True)[0])
        if fuller is not None:
            root -= fuller / (n - full.shape[1])
    else:
        root = kappa
    xt = x - root * project(x, full)
    left = xt.T @ x
    beta = np.linalg.solve(left, xt.T @ y)
    resid = y - x @ beta
    s2 = float(resid @ resid) / (n - x.shape[1])
    cov = s2 * np.linalg.inv(left)
    return np.asarray(beta, dtype=np.float64), np.asarray(np.sqrt(np.diag(cov)), dtype=np.float64), root


def tsls_case(name: str, seed: int, n: int, m: int, c: int, strength: float) -> Case:
    y, d, z, w = design(seed, n, m, c, strength)
    exog = sm.add_constant(w, prepend=False)  # pyright: ignore[reportUnknownMemberType]
    instruments = np.column_stack([z, exog])
    res = IV2SLS(y, np.column_stack([d, exog]), instruments).fit()
    full = np.column_stack([exog, z])
    ols = sm.OLS(d, full).fit()  # pyright: ignore[reportUnknownMemberType]
    restriction = np.zeros((m, full.shape[1]))
    restriction[np.arange(m), exog.shape[1] + np.arange(m)] = 1.0
    ftest = ols.f_test(restriction)  # pyright: ignore[reportUnknownMemberType]
    out: dict[str, FloatArray] = {
        "coefficients": np.asarray(res.params, dtype=np.float64),
        "standardErrors": np.asarray(res.bse, dtype=np.float64),
        "firstStageF": np.array(float(np.squeeze(ftest.fvalue))),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    }
    if m > 1:
        beta = np.asarray(res.params, dtype=np.float64)
        u = y - np.column_stack([d, exog]) @ beta
        ru = project(u.reshape(-1, 1), full).ravel()
        stat = n * (float(u @ u) - float(ru @ ru)) / float(u @ u)
        out["sargan"] = np.array([stat, float(chi2.sf(stat, m - 1))])
    return case(name, "tsls", [y, d, z, w], out, rtol=1e-8, atol=1e-9)


def liml_case(name: str, seed: int, n: int, m: int, c: int, strength: float, fuller: float | None) -> Case:
    y, d, z, w = design(seed, n, m, c, strength)
    beta, se, kappa = kclass(y, d, z, w, None, fuller)
    return case(
        name,
        "liml",
        [y, d, z, w],
        {"coefficients": beta, "standardErrors": se, "kappa": np.array(kappa)},
        rtol=1e-7,
        atol=1e-9,
        fuller=-1.0 if fuller is None else fuller,
    )


def cases() -> list[Case]:
    return [
        tsls_case("tsls/just-identified", 101, 400, 1, 0, 1.0),
        tsls_case("tsls/controls", 102, 500, 1, 2, 0.8),
        tsls_case("tsls/overidentified", 103, 600, 3, 2, 0.7),
        tsls_case("tsls/weak", 104, 400, 2, 1, 0.15),
        liml_case("liml/overidentified", 111, 500, 3, 1, 0.5, None),
        liml_case("liml/weak", 112, 400, 4, 0, 0.1, None),
        liml_case("fuller/one", 113, 500, 3, 1, 0.5, 1.0),
        liml_case("liml/just-identified", 114, 300, 1, 1, 1.0, None),
    ]
