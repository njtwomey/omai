"""The threshold sweep (analyseThresholds) against sklearn's per-threshold counts, ROC curve and average precision."""

import numpy as np
import sklearn
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def ascending(values: FloatArray, last: float) -> FloatArray:
    """sklearn's per-threshold values (thresholds decreasing) in the sweep's order: ascending, then +Infinity."""
    return np.concatenate([np.asarray(values, dtype=np.float64)[::-1], [last]])


def cases() -> list[Case]:
    g = rng(77)
    n = 70
    y = (g.uniform(size=n) > 0.6).astype(np.float64)
    s = 1.2 * y + g.normal(size=n)
    w = g.uniform(0.2, 2.0, size=n)
    out: list[Case] = []
    for label, scores, weights in [("continuous", s, None), ("tied", np.round(s, 1), None), ("weighted", s, w)]:
        inputs = [y, scores] if weights is None else [y, scores, weights]
        counts = M.confusion_matrix_at_thresholds(y, scores, sample_weight=weights)
        fps, tps, thr = counts[1], counts[3], counts[4]
        fpr, tpr, _ = M.roc_curve(y, scores, sample_weight=weights, drop_intermediate=False)
        out += [
            case(f"{label}/thresholds", "thresholds", inputs, ascending(thr, np.inf)),
            case(f"{label}/tp", "tp", inputs, ascending(tps, 0.0)),
            case(f"{label}/fp", "fp", inputs, ascending(fps, 0.0)),
            # roc_curve prepends the (0, 0) point at threshold +Infinity: the same order, reversed.
            case(f"{label}/fpr", "fpr", inputs, np.asarray(fpr, dtype=np.float64)[::-1].copy()),
            case(f"{label}/tpr", "tpr", inputs, np.asarray(tpr, dtype=np.float64)[::-1].copy()),
            case(f"{label}/rocAuc", "rocAuc", inputs, np.asarray(M.roc_auc_score(y, scores, sample_weight=weights))),
            case(
                f"{label}/averagePrecision",
                "averagePrecision",
                inputs,
                np.asarray(M.average_precision_score(y, scores, sample_weight=weights)),
            ),
        ]
    return out
