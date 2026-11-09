"""class_likelihood_ratios against sklearn.metrics: weighted and unweighted, and the ill-defined cases (no positives, no
predicted positives, no false positives, no true negatives) under every `replace_undefined_by`."""

import warnings

import numpy as np
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.exceptions import UndefinedMetricWarning

REFERENCE: dict[str, str] = {"scikit-learn": __import__("sklearn").__version__}


def cases() -> list[Case]:
    g = rng(404)
    n = 60
    t = (g.uniform(size=n) > 0.4).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.7, t, 1 - t)
    w = g.uniform(0.2, 2.0, size=n)
    problems: list[tuple[str, FloatArray, FloatArray]] = [
        ("random", t, p),
        ("no-positives", np.array([0.0, 0, 0, 0, 0]), np.array([0.0, 1, 0, 1, 0])),
        ("no-predicted-positives", np.array([1.0, 0, 1, 0, 1, 0]), np.zeros(6)),
        ("perfect", np.array([1.0, 0, 1, 0, 1, 0]), np.array([1.0, 0, 1, 0, 1, 0])),
        ("no-true-negatives", np.array([1.0, 0, 1, 1, 0, 1]), np.array([1.0, 1, 1, 0, 1, 1])),
    ]
    out: list[Case] = []
    replacements: list[tuple[str, object, float, float]] = [
        ("nan", np.nan, -1.0, -1.0),
        ("one", 1.0, -1.0, -1.0),
        ("pair", {"LR+": 5.0, "LR-": 0.25}, 5.0, 0.25),
    ]
    for name, yt, yp in problems:
        for weighted in [False, True]:
            if weighted and name != "random":
                continue
            for label, replace, pos, neg in replacements:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UndefinedMetricWarning)
                    lr_pos, lr_neg = M.class_likelihood_ratios(
                        yt,
                        yp,
                        sample_weight=w if weighted else None,
                        replace_undefined_by=replace,  # pyright: ignore[reportArgumentType]
                    )
                out.append(
                    case(
                        f"{name}/{'weighted' if weighted else 'plain'}/{label}",
                        "ratios",
                        [yt, yp, w],
                        {"positive": np.array(lr_pos), "negative": np.array(lr_neg)},
                        rtol=1e-12,
                        atol=1e-12,
                        replace=label,
                        weighted=weighted,
                        pairPositive=pos,
                        pairNegative=neg,
                    )
                )
    return out
