"""Precision-recall curves and average precision against sklearn.metrics; precision-recall-gain curves against the
authors' reference code (pyprg's prg.py by Kull and Flach, ported verbatim here because the package no longer runs
under numpy 2: `np.alen` became `len`, and the plotting is left out).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn
import sklearn.metrics as M
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "pyprg": "0.1.1b7 (ported)"}

type Points = dict[str, npt.NDArray[Any]]


def precision_gain(tp: FloatArray, fn: FloatArray, fp: FloatArray, tn: FloatArray) -> FloatArray:
    """prg.py's precision_gain on arrays (scalars go in as one-element arrays)."""
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - (n_pos / n_neg) * (fp / tp)
    return np.where(tn + fn == 0, 0.0, prec_gain)


def recall_gain(tp: FloatArray, fn: FloatArray, fp: FloatArray, tn: FloatArray) -> FloatArray:
    """prg.py's recall_gain on arrays."""
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        rg = 1.0 - (n_pos / n_neg) * (fn / tp)
    return np.where(tn + fn == 0, 1.0, rg)


def gain_of(
    f: Callable[[FloatArray, FloatArray, FloatArray, FloatArray], FloatArray], p: FloatArray, k: dict[str, int]
) -> float:
    """A gain of one point (the scalar branch of prg.py)."""

    def one(key: str) -> FloatArray:
        return np.asarray([p[k[key]]])

    return float(f(one("TP"), one("FN"), one("FP"), one("TN"))[0])


def create_segments(labels: FloatArray, pos_scores: FloatArray, neg_scores: FloatArray) -> Points:
    n = len(labels)
    new_order = np.lexsort((neg_scores, -pos_scores))
    labels, pos_scores, neg_scores = labels[new_order], pos_scores[new_order], neg_scores[new_order]
    segments: Points = {k: np.zeros(n) for k in ["pos_score", "neg_score", "pos_count", "neg_count"]}
    j = -1
    for i, label in enumerate(labels):
        if (i == 0) or (pos_scores[i - 1] != pos_scores[i]) or (neg_scores[i - 1] != neg_scores[i]):
            j += 1
            segments["pos_score"][j] = pos_scores[i]
            segments["neg_score"][j] = neg_scores[i]
        if label == 0:
            segments["neg_count"][j] += 1
        else:
            segments["pos_count"][j] += 1
    return {k: v[0 : j + 1] for k, v in segments.items()}


def get_point(points: Points, index: int) -> tuple[FloatArray, dict[str, int]]:
    keys = list(points.keys())
    point = np.zeros(len(keys))
    key_indices: dict[str, int] = {}
    for i, key in enumerate(keys):
        point[i] = points[key][index]
        key_indices[key] = i
    return point, key_indices


def insert_point(
    new_point: FloatArray,
    key_indices: dict[str, int],
    points: Points,
    precision_gain: float = 0,
    recall_gain: float = 0,
    is_crossing: float = 0,
) -> Points:
    for key in key_indices:
        points[key] = np.insert(points[key], 0, new_point[key_indices[key]])
    points["precision_gain"][0] = precision_gain
    points["recall_gain"][0] = recall_gain
    points["is_crossing"][0] = is_crossing
    new_order = np.lexsort((-points["precision_gain"], points["recall_gain"]))
    return {key: value[new_order] for key, value in points.items()}


def create_crossing_points(points: Points, n_pos: float, n_neg: float) -> Points:
    n = n_pos + n_neg
    points["is_crossing"] = np.zeros(len(points["pos_score"]))
    j = int(np.amin(np.where(points["recall_gain"] >= 0)[0]))
    if points["recall_gain"][j] > 0:
        point_1, key_indices_1 = get_point(points, j)
        point_2, _ = get_point(points, j - 1)
        delta = point_1 - point_2
        if delta[key_indices_1["TP"]] > 0:
            alpha = (n_pos * n_pos / n - points["TP"][j - 1]) / delta[key_indices_1["TP"]]
        else:
            alpha = 0.5
        with np.errstate(all="ignore"):
            new_point = point_2 + alpha * delta
        k = key_indices_1
        new_prec_gain = gain_of(precision_gain, new_point, k)
        points = insert_point(new_point, k, points, precision_gain=new_prec_gain, is_crossing=1)
    x, y = points["recall_gain"], points["precision_gain"]
    temp_y_0, temp_0_y, temp_1_x = np.append(y, 0), np.append(0, y), np.append(1, x)
    with np.errstate(invalid="ignore"):
        indices = np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]
    for i in indices:
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        point_1, key_indices_1 = get_point(points, int(i))
        point_2, _ = get_point(points, int(i) - 1)
        delta = point_1 - point_2
        k = key_indices_1
        if delta[k["TP"]] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - points["TP"][i - 1]) / delta[k["TP"]]
        else:
            alpha = (n_neg / n_pos * points["TP"][i - 1] - points["FP"][i - 1]) / delta[k["FP"]]
        with np.errstate(all="ignore"):
            new_point = point_2 + alpha * delta
        new_rec_gain = gain_of(recall_gain, new_point, k)
        points = insert_point(new_point, k, points, recall_gain=new_rec_gain, is_crossing=1)
        indices += 1
        x, y = points["recall_gain"], points["precision_gain"]
    return points


def create_prg_curve(labels: FloatArray, pos_scores: FloatArray) -> Points:
    neg_scores = -pos_scores
    n_pos = float(np.sum(labels))
    n_neg = len(labels) - n_pos
    labels = 1 * (labels == 1)
    segments = create_segments(labels, pos_scores, neg_scores)
    points: Points = {
        "pos_score": np.insert(segments["pos_score"], 0, np.inf),
        "neg_score": np.insert(segments["neg_score"], 0, -np.inf),
        "TP": np.insert(np.cumsum(segments["pos_count"]), 0, 0),
        "FP": np.insert(np.cumsum(segments["neg_count"]), 0, 0),
    }
    points["FN"] = n_pos - points["TP"]
    points["TN"] = n_neg - points["FP"]
    points["precision_gain"] = precision_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    points["recall_gain"] = recall_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    return create_crossing_points(points, n_pos, n_neg)


def calc_auprg(prg_curve: Points) -> float:
    area = 0.0
    rg, pg = prg_curve["recall_gain"], prg_curve["precision_gain"]
    for i in range(1, len(rg)):
        if (not np.isnan(rg[i - 1])) and (rg[i - 1] >= 0):
            area += (rg[i] - rg[i - 1]) * (pg[i] + pg[i - 1]) / 2
    return area


def cases() -> list[Case]:
    g = rng(55)
    n = 60
    y = (g.uniform(size=n) > 0.6).astype(np.float64)
    s = y + g.normal(size=n)
    s_tied = np.round(s, 1)
    w = g.uniform(0.2, 2.0, size=n)
    yk = g.integers(0, 3, size=n).astype(np.float64)
    logits = g.normal(size=(n, 3)) + 1.5 * np.eye(3)[yk.astype(int)]
    ym = (g.uniform(size=(n, 3)) > 0.5).astype(np.float64)
    sm = ym + g.normal(size=(n, 3))
    out: list[Case] = []
    for label, scores in [("continuous", s), ("tied", s_tied)]:
        for drop in [False, True]:
            p, r, t = M.precision_recall_curve(y, scores, drop_intermediate=drop)
            out += [
                case(f"prCurve/{label}/drop={drop}/precision", "prPrecision", [y, scores], p, dropIntermediate=drop),
                case(f"prCurve/{label}/drop={drop}/recall", "prRecall", [y, scores], r, dropIntermediate=drop),
                case(f"prCurve/{label}/drop={drop}/thresholds", "prThresholds", [y, scores], t, dropIntermediate=drop),
            ]
        out.append(
            case(
                f"averagePrecisionScore/{label}",
                "averagePrecisionScore",
                [y, scores],
                np.asarray(M.average_precision_score(y, scores)),
            )
        )
        curve = create_prg_curve(y, scores)
        out += [
            case(f"prGainCurve/{label}/recallGain", "prgRecallGain", [y, scores], curve["recall_gain"]),
            case(f"prGainCurve/{label}/precisionGain", "prgPrecisionGain", [y, scores], curve["precision_gain"]),
            case(f"prGainAuc/{label}", "prGainAuc", [y, scores], np.asarray(calc_auprg(curve))),
        ]
    out += [
        case(
            "averagePrecisionScore/weighted",
            "averagePrecisionScore",
            [y, s, w],
            np.asarray(M.average_precision_score(y, s, sample_weight=w)),
        ),
        case(
            "averagePrecisionScore/multiclass-macro",
            "averagePrecisionScore",
            [yk, logits],
            np.asarray(M.average_precision_score(yk, logits)),
        ),
        case(
            "averagePrecisionScore/multiclass-weighted",
            "averagePrecisionScore",
            [yk, logits, w],
            np.asarray(M.average_precision_score(yk, logits, average="weighted", sample_weight=w)),
            average="weighted",
        ),
        case(
            "averagePrecisionScore/multilabel-micro",
            "averagePrecisionScore",
            [ym, sm],
            np.asarray(M.average_precision_score(ym, sm, average="micro")),
            average="micro",
        ),
        case(
            "averagePrecisionScore/multilabel-samples",
            "averagePrecisionScore",
            [ym, sm],
            np.asarray(M.average_precision_score(ym, sm, average="samples")),
            average="samples",
        ),
        case(
            "prCurve/weighted/precision", "prPrecision", [y, s, w], M.precision_recall_curve(y, s, sample_weight=w)[0]
        ),
    ]
    return out
