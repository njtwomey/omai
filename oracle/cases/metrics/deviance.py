"""Deviance, pinball and D^2 metrics against sklearn.metrics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

METRICS: list[tuple[str, Callable[..., Any], dict[str, Any], bool, bool]] = [
    ("meanPinballLoss", M.mean_pinball_loss, {"alpha": 0.8}, False, True),
    ("d2AbsoluteErrorScore", M.d2_absolute_error_score, {}, False, True),
    ("d2PinballScore", M.d2_pinball_score, {"alpha": 0.8}, False, True),
    ("d2PinballScore", M.d2_pinball_score, {"alpha": 0.25}, False, True),
    ("meanPoissonDeviance", M.mean_poisson_deviance, {}, True, False),
    ("meanGammaDeviance", M.mean_gamma_deviance, {}, True, False),
    ("meanTweedieDeviance", M.mean_tweedie_deviance, {"power": 1.5}, True, False),
    ("meanTweedieDeviance", M.mean_tweedie_deviance, {"power": 3}, True, False),
    ("meanTweedieDeviance", M.mean_tweedie_deviance, {"power": -0.5}, False, False),
    ("d2TweedieScore", M.d2_tweedie_score, {"power": 0}, False, False),
    ("d2TweedieScore", M.d2_tweedie_score, {"power": 1}, True, False),
    ("d2TweedieScore", M.d2_tweedie_score, {"power": 2}, True, False),
    ("d2TweedieScore", M.d2_tweedie_score, {"power": 1.4}, True, False),
]


def cases() -> list[Case]:
    g = rng(53)
    y = g.normal(size=20)
    p = y + 0.5 * g.normal(size=20)
    ypos = np.exp(g.normal(size=20))
    ppos = ypos * np.exp(0.3 * g.normal(size=20))
    w = g.uniform(0.2, 2, size=20)
    y2, p2 = g.normal(size=(15, 3)), g.normal(size=(15, 3))
    tie = np.array([1.0, 2.0, 3.0, 4.0])
    out: list[Case] = []
    for name, f, kw, positive, multi in METRICS:
        t, q = (ypos, ppos) if positive else (y, np.exp(p) if kw.get("power", 0) < 0 else p)
        label = name + "".join(f"/{k}={v}" for k, v in kw.items())
        out.append(case(label, name, [t, q], np.array(f(t, q, **kw)), **kw))
        out.append(case(f"{label}/weighted", name, [t, q, w], np.array(f(t, q, sample_weight=w, **kw)), **kw))
        if multi:
            out.append(case(f"{label}/multioutput", name, [y2, p2], np.array(f(y2, p2, **kw)), **kw))
    # An even sample whose median falls between two values (sklearn averages at exact ties).
    out.append(
        case(
            "d2AbsoluteErrorScore/tie",
            "d2AbsoluteErrorScore",
            [tie, tie[::-1]],
            np.array(M.d2_absolute_error_score(tie, tie[::-1])),
        )
    )
    return out
