"""Regression metrics against sklearn.metrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

METRICS: list[tuple[str, Callable[..., Any], dict[str, Any], bool]] = [
    ("meanSquaredError", M.mean_squared_error, {}, False),
    ("rootMeanSquaredError", M.root_mean_squared_error, {}, False),
    ("meanAbsoluteError", M.mean_absolute_error, {}, False),
    ("medianAbsoluteError", M.median_absolute_error, {}, False),
    ("r2Score", M.r2_score, {}, False),
    ("explainedVarianceScore", M.explained_variance_score, {}, False),
    ("meanAbsolutePercentageError", M.mean_absolute_percentage_error, {}, False),
    ("meanPinballLoss", M.mean_pinball_loss, {"alpha": 0.8}, False),
    ("d2AbsoluteErrorScore", M.d2_absolute_error_score, {}, False),
    ("meanSquaredLogError", M.mean_squared_log_error, {}, True),
    ("meanPoissonDeviance", M.mean_poisson_deviance, {}, True),
    ("meanGammaDeviance", M.mean_gamma_deviance, {}, True),
    ("meanTweedieDeviance", M.mean_tweedie_deviance, {"power": 1.5}, True),
    ("meanTweedieDeviance", M.mean_tweedie_deviance, {"power": 3}, True),
]


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
    return out
