"""Calibration against sklearn.calibration: Platt scaling, isotonic regression, temperature scaling, beta calibration
(as a logistic regression on log features), vector scaling (scipy's L-BFGS on the log loss), histogram binning (a
numpy reference) and CalibratedClassifierCV with ensembles and out-of-fold calibration."""

from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.calibration import CalibratedClassifierCV, _sigmoid_calibration, _TemperatureScaling  # pyright: ignore[reportPrivateUsage]
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def vector_scaling(z: FloatArray, y: FloatArray) -> FloatArray:
    """Fit softmax(w z + b) by L-BFGS on the mean log loss; return the fitted scales and biases."""
    k = z.shape[1]
    onehot = np.eye(k)[y.astype(int)]

    def nll(p: FloatArray) -> float:
        return float(-(onehot * log_softmax(z * p[:k] + p[k:], axis=1)).sum() / len(y))

    def grad(p: FloatArray) -> FloatArray:
        r = (softmax(z * p[:k] + p[k:], axis=1) - onehot) / len(y)
        return np.concatenate([(r * z).sum(0), r.sum(0)])

    res = minimize(
        nll,
        np.concatenate([np.ones(k), np.zeros(k)]),
        jac=grad,
        method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000},
    )
    return np.asarray(res.x)


def histogram(scores: FloatArray, y: FloatArray, test: FloatArray, n_bins: int) -> FloatArray:
    """Equal-width bins on [0, 1]; each bin's positive rate (the overall rate for empty bins)."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges[1:-1], scores, side="right"), 0, n_bins - 1)
    rate = np.array([y[bins == b].mean() if np.any(bins == b) else y.mean() for b in range(n_bins)])
    return rate[np.clip(np.searchsorted(edges[1:-1], test, side="right"), 0, n_bins - 1)]


def cases() -> list[Case]:
    g = rng(4341)
    n = 60
    s = g.normal(size=n)
    y = (s + 0.8 * g.normal(size=n) > 0).astype(np.float64)
    test = np.concatenate([np.linspace(-3, 3, 9), [s[3], s[10]]])
    out: list[Case] = []
    a, b = _sigmoid_calibration(s, y)
    out.append(
        case(
            "platt",
            "platt",
            [s, y, test],
            {"a": np.array(a), "b": np.array(b), "predict": 1 / (1 + np.exp(a * test + b))},
            rtol=1e-5,
            atol=1e-7,
        )
    )
    iso = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    out.append(case("isotonic", "isotonic", [s, y, test], {"predict": iso.predict(test)}, rtol=1e-12, atol=1e-12))
    sr = np.round(s, 1)
    iso = IsotonicRegression(out_of_bounds="clip").fit(sr, y)
    out.append(case("isotonic/ties", "isotonic", [sr, y, test], {"predict": iso.predict(test)}, rtol=1e-12, atol=1e-12))
    ts: Any = _TemperatureScaling().fit(s, y)
    out.append(
        case(
            "temperature/binary",
            "temperature",
            [s, y, test],
            {"beta": np.array(ts.beta_), "predict": ts.predict(test)},
            rtol=1e-6,
            atol=1e-8,
        )
    )
    z = g.normal(size=(n, 3)) * 2
    y3 = np.array([int(np.argmax(r + 1.5 * g.normal(size=3))) for r in z], dtype=np.float64)
    ts = _TemperatureScaling().fit(z, y3)
    out.append(
        case(
            "temperature/multiclass",
            "temperature",
            [z, y3, z[:5]],
            {"beta": np.array(ts.beta_), "predict": ts.predict(z[:5])},
            rtol=1e-6,
            atol=1e-8,
        )
    )
    p = vector_scaling(z, y3)
    out.append(
        case(
            "vector",
            "vector",
            [z, y3, z[:5]],
            {"predict": softmax(z[:5] * p[:3] + p[3:], axis=1)},
            rtol=1e-5,
            atol=1e-7,
        )
    )
    prob = 1 / (1 + np.exp(-1.7 * s + 0.3))
    feats = np.column_stack([np.log(prob), -np.log1p(-prob)])
    lr = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000).fit(feats, y)
    tp = 1 / (1 + np.exp(-1.7 * test + 0.3))
    tf = np.column_stack([np.log(tp), -np.log1p(-tp)])
    out.append(
        case(
            "beta",
            "beta",
            [prob, y, tp],
            {
                "a": np.array(lr.coef_[0, 0]),
                "b": np.array(lr.coef_[0, 1]),
                "c": np.array(lr.intercept_[0]),
                "predict": lr.predict_proba(tf)[:, 1],
            },
            rtol=1e-5,
            atol=1e-7,
        )
    )
    out.append(case("histogram", "histogram", [prob, y, tp], {"predict": histogram(prob, y, tp, 5)}, nBins=5))
    x = np.column_stack([s, g.normal(size=n)])
    xt = np.column_stack([test, np.zeros(len(test))])
    for method in ["sigmoid", "isotonic", "temperature"]:
        for ensemble in [True, False]:
            # sklearn types ensemble as str ("auto"); a bool is accepted.
            options: dict[str, Any] = {"method": method, "cv": StratifiedKFold(3), "ensemble": ensemble}
            cal = CalibratedClassifierCV(GaussianNB(), **options)
            cal.fit(x, y)
            out.append(
                case(
                    f"cv/{method}/{ensemble}",
                    "calibratedCv",
                    [x, y, xt],
                    {"proba": cal.predict_proba(xt)},
                    method=method,
                    ensemble=ensemble,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )
    x3 = np.column_stack([z[:, :2], g.normal(size=n)])
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(GaussianNB(), method=method, cv=StratifiedKFold(3)).fit(x3, y3)
        out.append(
            case(
                f"cv/multiclass/{method}",
                "calibratedCv",
                [x3, y3, x3[:6]],
                {"proba": cal.predict_proba(x3[:6])},
                method=method,
                ensemble=True,
                rtol=1e-5,
                atol=1e-7,
            )
        )
    lr = LogisticRegression(tol=1e-12, max_iter=10000)
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv=StratifiedKFold(3)).fit(x, y)
    out.append(
        case(
            "cv/logistic-decision",
            "calibratedLogistic",
            [x, y, xt],
            {"proba": cal.predict_proba(xt)},
            rtol=1e-5,
            atol=1e-7,
        )
    )
    return out
