"""Treatment-effect estimators against prebuilt references: statsmodels OLS with HC1 errors for regression adjustment
and GLM with an offset for the TMLE fluctuation, scikit-learn's logistic and linear regressions for the nuisance models
of inverse probability weighting, AIPW, stratification and TMLE, scikit-learn's nearest neighbours for matching and
scipy's BFGS on the dual for entropy balancing. The estimators' formulas are written out in numpy beside them."""

import numpy as np
import numpy.typing as npt
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

REFERENCE: dict[str, str] = {
    "statsmodels": statsmodels.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
}

CLIP = 0.01


def study(seed: int, n: int, d: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Confounded observational data with a constant effect of 2 and a nonlinear nuisance."""
    g = rng(seed)
    x = g.normal(size=(n, d))
    e = 1.0 / (1.0 + np.exp(-(0.8 * x[:, 0] - 0.5 * x[:, 1])))
    t = (g.uniform(size=n) < e).astype(np.float64)
    y = 1.0 + x[:, 0] + 0.5 * x[:, 1] ** 2 + 2.0 * t + 0.7 * g.normal(size=n)
    return x, t, y


def propensity(x: FloatArray, t: FloatArray) -> FloatArray:
    model = LogisticRegression(C=np.inf, tol=1e-12, max_iter=5000).fit(x, t)  # pyright: ignore[reportUnknownMemberType]
    p = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
    return np.clip(p, CLIP, 1 - CLIP)


def outcome_models(x: FloatArray, t: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    out: list[FloatArray] = []
    for arm in (0.0, 1.0):
        m = LinearRegression().fit(x[t == arm], y[t == arm])  # pyright: ignore[reportUnknownMemberType]
        out.append(np.asarray(m.predict(x), dtype=np.float64))  # pyright: ignore[reportUnknownMemberType]
    return out[0], out[1]


def hajek(t: FloatArray, y: FloatArray, w: FloatArray) -> tuple[float, FloatArray]:
    n = len(t)
    s1 = float(np.sum(t * w))
    s0 = float(np.sum((1 - t) * w))
    m1 = float(np.sum(t * w * y)) / s1
    m0 = float(np.sum((1 - t) * w * y)) / s0
    infl = np.where(t == 1, n * w * (y - m1) / s1, -n * w * (y - m0) / s0)
    return m1 - m0, np.asarray(infl, dtype=np.float64)


def ipw_case(name: str, seed: int, estimand: str) -> Case:
    x, t, y = study(seed, 400, 3)
    e = propensity(x, t)
    w = np.where(t == 1, 1 / e, 1 / (1 - e)) if estimand == "ate" else np.where(t == 1, 1.0, e / (1 - e))
    value, infl = hajek(t, y, np.asarray(w, dtype=np.float64))
    se = float(np.sqrt(np.var(infl, ddof=1) / len(t)))
    ess = float(np.sum(w) ** 2 / np.sum(w**2))
    return case(
        name,
        "ipw",
        [x, t, y],
        {"value": np.array(value), "se": np.array(se), "weights": np.asarray(w, dtype=np.float64)},
        rtol=1e-5,
        atol=1e-7,
        estimand=estimand,
        ess=ess,
    )


def aipw_case(name: str, seed: int, estimand: str) -> Case:
    x, t, y = study(seed, 400, 3)
    e = propensity(x, t)
    m0, m1 = outcome_models(x, t, y)
    n = len(t)
    if estimand == "ate":
        gamma = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        value = float(np.mean(gamma))
        infl = gamma - value
    else:
        n1 = float(np.sum(t))
        r = y - m0
        value = float(np.sum(t * r - (1 - t) * e / (1 - e) * r) / n1)
        infl = (t * (r - value) - (1 - t) * e / (1 - e) * r) / (n1 / n)
    se = float(np.sqrt(np.var(infl, ddof=1) / n))
    return case(
        name,
        "aipw",
        [x, t, y],
        {"value": np.array(value), "se": np.array(se)},
        rtol=1e-5,
        atol=1e-7,
        estimand=estimand,
    )


def stratification_case(name: str, seed: int, strata: int) -> Case:
    x, t, y = study(seed, 500, 3)
    e = propensity(x, t)
    order = np.argsort(e, kind="stable")
    n = len(t)
    effects: list[float] = []
    weights: list[float] = []
    variances: list[float] = []
    for s in range(strata):
        rows = order[(s * n) // strata : ((s + 1) * n) // strata]
        y1, y0 = y[rows][t[rows] == 1], y[rows][t[rows] == 0]
        if len(y1) == 0 or len(y0) == 0:
            continue
        effects.append(float(np.mean(y1) - np.mean(y0)))
        weights.append(float(len(rows)))
        v1 = float(np.var(y1, ddof=1)) / len(y1) if len(y1) > 1 else 0.0
        v0 = float(np.var(y0, ddof=1)) / len(y0) if len(y0) > 1 else 0.0
        variances.append(v1 + v0)
    w = np.array(weights) / sum(weights)
    value = float(np.sum(w * np.array(effects)))
    se = float(np.sqrt(np.sum(w**2 * np.array(variances))))
    return case(
        name,
        "stratification",
        [x, t, y],
        {"value": np.array(value), "se": np.array(se)},
        rtol=1e-5,
        atol=1e-7,
        strata=strata,
    )


def matching_case(name: str, seed: int, on: str, neighbours: int) -> Case:
    x, t, y = study(seed, 300, 3)
    treated, control = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    if on == "propensity":
        e = propensity(x, t)
        z = np.log(e / (1 - e)).reshape(-1, 1)
    else:
        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    nn = NearestNeighbors(n_neighbors=neighbours).fit(z[control])  # pyright: ignore[reportUnknownMemberType]
    _, idx = nn.kneighbors(z[treated])  # pyright: ignore[reportUnknownMemberType]
    imputed = y[control][idx].mean(axis=1)
    value = float(np.mean(y[treated] - imputed))
    return case(
        name,
        "matching",
        [x, t, y],
        {"value": np.array(value)},
        rtol=1e-5,
        atol=1e-7,
        on=on,
        neighbours=neighbours,
    )


def entropy_case(name: str, seed: int) -> Case:
    x, t, y = study(seed, 400, 3)
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    target = z[t == 1].mean(axis=0)
    c = z[t == 0] - target

    def dual(lam: npt.NDArray[np.float64]) -> float:
        a = -(c @ lam)
        top = float(a.max())
        return top + float(np.log(np.exp(a - top).sum()))

    def grad(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a = -(c @ lam)
        w = np.exp(a - a.max())
        w /= w.sum()
        return np.asarray(-(w @ c), dtype=np.float64)

    res = minimize(dual, np.zeros(c.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-12})  # pyright: ignore[reportUnknownMemberType]
    a = -(c @ res.x)
    w = np.exp(a - a.max())
    w /= w.sum()
    value = float(np.mean(y[t == 1]) - np.sum(w * y[t == 0]))
    return case(
        name,
        "entropyBalancing",
        [x, t, y],
        {"value": np.array(value), "controlWeights": np.asarray(w, dtype=np.float64)},
        rtol=1e-6,
        atol=1e-9,
    )


def expit(z: FloatArray) -> FloatArray:
    return np.asarray(1 / (1 + np.exp(-z)), dtype=np.float64)


def tmle_case(name: str, seed: int) -> Case:
    x, t, y = study(seed, 400, 3)
    e = propensity(x, t)
    m0, m1 = outcome_models(x, t, y)
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    ys = (y - lo) / span
    q0 = np.clip((m0 - lo) / span, 1e-4, 1 - 1e-4)
    q1 = np.clip((m1 - lo) / span, 1e-4, 1 - 1e-4)
    qa = np.where(t == 1, q1, q0)
    h = np.where(t == 1, 1 / e, -1 / (1 - e))
    offset = np.log(qa / (1 - qa))
    fit = sm.GLM(ys, h.reshape(-1, 1), family=sm.families.Binomial(), offset=offset).fit(tol=1e-12, maxiter=200)  # pyright: ignore[reportUnknownMemberType]
    eps = float(fit.params[0])
    q1s = expit(np.log(q1 / (1 - q1)) + eps / e)
    q0s = expit(np.log(q0 / (1 - q0)) - eps / (1 - e))
    psi = float(np.mean(q1s - q0s))
    qas = np.where(t == 1, q1s, q0s)
    infl = span * (h * (ys - qas) + q1s - q0s - psi)
    se = float(np.sqrt(np.var(infl, ddof=1) / len(t)))
    return case(
        name,
        "tmle",
        [x, t, y],
        {"epsilon": np.array(eps), "value": np.array(span * psi), "se": np.array(se)},
        rtol=1e-5,
        atol=1e-8,
    )


def regression_case(name: str, seed: int) -> Case:
    x, t, y = study(seed, 300, 3)
    design = sm.add_constant(np.column_stack([t, x]))  # pyright: ignore[reportUnknownMemberType]
    fit = sm.OLS(y, design).fit(cov_type="HC1")  # pyright: ignore[reportUnknownMemberType]
    return case(
        name,
        "regressionAdjustment",
        [x, t, y],
        {"value": np.array(float(fit.params[1])), "se": np.array(float(fit.bse[1]))},
        rtol=1e-8,
        atol=1e-10,
    )


def cases() -> list[Case]:
    return [
        regression_case("regression/a", 11),
        regression_case("regression/b", 12),
        ipw_case("ipw/ate", 21, "ate"),
        ipw_case("ipw/att", 22, "att"),
        aipw_case("aipw/ate", 31, "ate"),
        aipw_case("aipw/att", 32, "att"),
        stratification_case("stratification/five", 41, 5),
        stratification_case("stratification/ten", 42, 10),
        matching_case("matching/covariates", 51, "covariates", 1),
        matching_case("matching/propensity", 52, "propensity", 3),
        entropy_case("entropy/a", 61),
        entropy_case("entropy/b", 62),
        tmle_case("tmle/a", 71),
        tmle_case("tmle/b", 72),
    ]
