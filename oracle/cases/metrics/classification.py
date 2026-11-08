"""Classification metrics against sklearn.metrics (and imbalanced-learn's definitions of specificity)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# sklearn types pos_label as an int; a string label passes through a mapping instead.
POS_B: dict[str, Any] = {"pos_label": "b"}

AVERAGED: list[tuple[str, Callable[..., Any]]] = [
    ("precisionScore", M.precision_score),
    ("recallScore", M.recall_score),
    ("f1Score", M.f1_score),
    ("jaccardScore", M.jaccard_score),
]


def one_vs_rest(t: FloatArray, p: FloatArray, w: FloatArray | None) -> FloatArray:
    """Per-class [[tn, fp], [fn, tp]] from sklearn's multilabel_confusion_matrix."""
    return np.asarray(M.multilabel_confusion_matrix(t, p, sample_weight=w), dtype=np.float64)


def specificity(t: FloatArray, p: FloatArray, w: FloatArray | None, average: str) -> float:
    """tn / (tn + fp) from sklearn's one-vs-rest confusion matrices, averaged as imbalanced-learn does."""
    m = one_vs_rest(t, p, w)
    tn, fp, fn, tp = m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1]
    if average == "micro":
        return float(tn.sum() / (tn.sum() + fp.sum()))
    per = tn / (tn + fp)
    if average == "weighted":
        return float(np.average(per, weights=tp + fn))
    return float(per.mean())


def npv(t: FloatArray, p: FloatArray, w: FloatArray | None) -> float:
    m = one_vs_rest(t, p, w)
    tn, fn = m[:, 0, 0], m[:, 1, 0]
    return float((tn / (tn + fn)).mean())


def cases() -> list[Case]:
    g = rng(51)
    n = 40
    t = g.integers(0, 4, size=n).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.6, t, g.integers(0, 4, size=n)).astype(np.float64)
    w = g.uniform(0.2, 2.0, size=n)
    tb = (g.uniform(size=n) > 0.4).astype(np.float64)
    pb = np.where(g.uniform(size=n) < 0.7, tb, 1 - tb)
    tm = (g.uniform(size=(n, 3)) > 0.5).astype(np.float64)
    pm = np.where(g.uniform(size=(n, 3)) < 0.75, tm, 1 - tm)
    out: list[Case] = []
    for name, f in AVERAGED:
        out.append(case(f"{name}/binary", name, [tb, pb], np.asarray(f(tb, pb))))
        out.append(case(f"{name}/binary-weighted", name, [tb, pb, w], np.asarray(f(tb, pb, sample_weight=w))))
        out.append(case(f"{name}/posLabel0", name, [tb, pb], np.asarray(f(tb, pb, pos_label=0)), posLabel=0))
        for average in ["micro", "macro", "weighted"]:
            out.append(case(f"{name}/{average}", name, [t, p], np.asarray(f(t, p, average=average)), average=average))
            out.append(
                case(
                    f"{name}/{average}-weighted",
                    name,
                    [t, p, w],
                    np.asarray(f(t, p, average=average, sample_weight=w)),
                    average=average,
                )
            )
            out.append(
                case(
                    f"{name}/multilabel-{average}",
                    name,
                    [tm, pm],
                    np.asarray(f(tm, pm, average=average)),
                    average=average,
                )
            )
        out.append(
            case(
                f"{name}/samples",
                name,
                [tm, pm, w],
                np.asarray(f(tm, pm, average="samples", sample_weight=w)),
                average="samples",
            )
        )
        out.append(
            case(
                f"{name}/labels-subset",
                name,
                [t, p],
                np.asarray(f(t, p, labels=[1, 3], average="micro")),
                labels=[1, 3],
                average="micro",
            )
        )
    out += [
        case(
            "fbetaScore/2-macro",
            "fbetaScore",
            [t, p],
            np.asarray(M.fbeta_score(t, p, beta=2, average="macro")),
            beta=2,
            average="macro",
        ),
        case("fbetaScore/0.5-binary", "fbetaScore", [tb, pb], np.asarray(M.fbeta_score(tb, pb, beta=0.5)), beta=0.5),
        case("accuracyScore", "accuracyScore", [t, p], np.asarray(M.accuracy_score(t, p))),
        case("accuracyScore/weighted", "accuracyScore", [t, p, w], np.asarray(M.accuracy_score(t, p, sample_weight=w))),
        case(
            "accuracyScore/count",
            "accuracyScore",
            [t, p],
            np.asarray(M.accuracy_score(t, p, normalize=False)),
            normalize=False,
        ),
        case("accuracyScore/subset", "accuracyScore", [tm, pm], np.asarray(M.accuracy_score(tm, pm))),
        case("balancedAccuracyScore", "balancedAccuracyScore", [t, p], np.asarray(M.balanced_accuracy_score(t, p))),
        case(
            "balancedAccuracyScore/adjusted-weighted",
            "balancedAccuracyScore",
            [t, p, w],
            np.asarray(M.balanced_accuracy_score(t, p, sample_weight=w, adjusted=True)),
            adjusted=True,
        ),
        case(
            "specificityScore/macro",
            "specificityScore",
            [t, p],
            np.asarray(specificity(t, p, None, "macro")),
            average="macro",
        ),
        case(
            "specificityScore/weighted",
            "specificityScore",
            [t, p, w],
            np.asarray(specificity(t, p, w, "weighted")),
            average="weighted",
        ),
        case(
            "specificityScore/micro",
            "specificityScore",
            [t, p],
            np.asarray(specificity(t, p, None, "micro")),
            average="micro",
        ),
        case(
            "negativePredictiveValue/macro",
            "negativePredictiveValue",
            [t, p],
            np.asarray(npv(t, p, None)),
            average="macro",
        ),
        case("confusionMatrix", "confusionMatrix", [t, p], np.asarray(M.confusion_matrix(t, p), dtype=np.float64)),
        case(
            "confusionMatrix/weighted-true",
            "confusionMatrix",
            [t, p, w],
            np.asarray(M.confusion_matrix(t, p, sample_weight=w, normalize="true")),
            normalize="true",
        ),
        case(
            "confusionMatrix/pred",
            "confusionMatrix",
            [t, p],
            np.asarray(M.confusion_matrix(t, p, normalize="pred")),
            normalize="pred",
        ),
        case(
            "confusionMatrix/all-labels",
            "confusionMatrix",
            [t, p],
            np.asarray(M.confusion_matrix(t, p, labels=[3, 1, 0], normalize="all")),
            labels=[3, 1, 0],
            normalize="all",
        ),
        case("multilabelConfusionMatrix", "multilabelConfusionMatrix", [t, p], one_vs_rest(t, p, None)),
        case("multilabelConfusionMatrix/multilabel", "multilabelConfusionMatrix", [tm, pm, w], one_vs_rest(tm, pm, w)),
        case(
            "multilabelConfusionMatrix/samplewise",
            "multilabelConfusionMatrix",
            [tm, pm],
            np.asarray(M.multilabel_confusion_matrix(tm, pm, samplewise=True), dtype=np.float64),
            samplewise=True,
        ),
    ]
    s_true = ["cat", "dog", "bird", "dog", "cat", "bird", "dog"]
    s_pred = ["cat", "bird", "bird", "dog", "dog", "bird", "dog"]
    out.append(
        case(
            "f1Score/strings",
            "f1Score",
            [],
            np.asarray(M.f1_score(s_true, s_pred, average="weighted")),
            yTrue=s_true,
            yPred=s_pred,
            average="weighted",
        )
    )
    out.append(
        case(
            "precisionScore/strings-posLabel",
            "precisionScore",
            [],
            np.asarray(M.precision_score(["a", "b", "b"], ["b", "b", "a"], **POS_B)),
            yTrue=["a", "b", "b"],
            yPred=["b", "b", "a"],
            posLabel="b",
        )
    )
    return out
