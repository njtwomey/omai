"""Regression metrics against sklearn.metrics (and the adjusted R^2, log-cosh and Huber formulas over them)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

METRICS: list[tuple[str, Callable[..., Any], dict[str, Any], bool]] = [
    ("meanSquaredError", M.mean_squared_error, {}, False),
    ("rootMeanSquaredError", M.root_mean_squared_error, {}, False),
    ("meanAbsoluteError", M.mean_absolute_error, {}, False),
    ("medianAbsoluteError", M.median_absolute_error, {}, False),
    ("r2Score", M.r2_score, {}, False),
    ("explainedVarianceScore", M.explained_variance_score, {}, False),
    ("meanAbsolutePercentageError", M.mean_absolute_percentage_error, {}, False),
    ("meanSquaredLogError", M.mean_squared_log_error, {}, True),
]


def adjusted_r2(t: FloatArray, p: FloatArray, n_features: int, w: FloatArray | None = None) -> float:
    n = len(t) if w is None else float(w.sum())
    return 1 - (1 - float(M.r2_score(t, p, sample_weight=w))) * (n - 1) / (n - n_features - 1)


def huber(r: FloatArray, delta: float) -> FloatArray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta))


def cases() -> list[Case]:
    g = rng(50)
    y = g.normal(size=20)
    p = y + 0.5 * g.normal(size=20)
    ypos = np.exp(g.normal(size=20))
    ppos = ypos * np.exp(0.3 * g.normal(size=20))
    w = g.uniform(0.2, 2, size=20)
    y2, p2 = g.normal(size=(15, 3)), g.normal(size=(15, 3))
    out: list[Case] = []
    for name, f, kw, positive in METRICS:
        t, q = (ypos, ppos) if positive else (y, p)
        label = name + "".join(f"/{k}={v}" for k, v in kw.items())
        out.append(case(label, name, [t, q], np.array(f(t, q, **kw)), **kw))
        out.append(case(f"{label}/weighted", name, [t, q, w], np.array(f(t, q, sample_weight=w, **kw)), **kw))
        if not positive and name != "medianAbsoluteError":
            out.append(case(f"{label}/multioutput", name, [y2, p2], np.array(f(y2, p2, **kw)), **kw))
    out.append(case("maxError", "maxError", [y, p], np.array(M.max_error(y, p))))
    const = np.full(5, 2.0)
    out.append(case("r2Score/constant", "r2Score", [const, const + 1], np.array(M.r2_score(const, const + 1))))
    out += [
        case("adjustedR2Score", "adjustedR2Score", [y, p], np.asarray(adjusted_r2(y, p, 3)), nFeatures=3),
        case(
            "adjustedR2Score/weighted", "adjustedR2Score", [y, p, w], np.asarray(adjusted_r2(y, p, 2, w)), nFeatures=2
        ),
        case("meanLogCoshError", "meanLogCoshError", [y, 4 * p], np.asarray(np.log(np.cosh(y - 4 * p)).mean())),
        case(
            "meanLogCoshError/weighted",
            "meanLogCoshError",
            [y, p, w],
            np.asarray(np.average(np.log(np.cosh(y - p)), weights=w)),
        ),
        case("meanHuberLoss", "meanHuberLoss", [y, 2 * p], np.asarray(huber(y - 2 * p, 0.8).mean()), delta=0.8),
        case(
            "meanHuberLoss/weighted", "meanHuberLoss", [y, p, w], np.asarray(np.average(huber(y - p, 1.0), weights=w))
        ),
    ]
    return out
