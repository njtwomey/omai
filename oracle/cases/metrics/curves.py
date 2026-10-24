"""ROC and DET curves and ROC AUC against sklearn.metrics."""

import numpy as np
import sklearn
import sklearn.metrics as M
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(54)
    n = 60
    y = (g.uniform(size=n) > 0.55).astype(np.float64)
    s = y + g.normal(size=n)
    s_tied = np.round(s, 1)
    w = g.uniform(0.2, 2.0, size=n)
    yk = g.integers(0, 3, size=n).astype(np.float64)
    logits = g.normal(size=(n, 3)) + 1.5 * np.eye(3)[yk.astype(int)]
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ym = (g.uniform(size=(n, 3)) > 0.5).astype(np.float64)
    sm = ym + g.normal(size=(n, 3))
    out: list[Case] = []
    for label, scores in [("continuous", s), ("tied", s_tied)]:
        c = M.confusion_matrix_at_thresholds(y, scores)
        out += [
            case(f"confusionMatrixAtThresholds/{label}/tps", "thresholdTps", [y, scores], np.asarray(c[3])),
            case(f"confusionMatrixAtThresholds/{label}/fps", "thresholdFps", [y, scores], np.asarray(c[1])),
            case(f"confusionMatrixAtThresholds/{label}/thresholds", "thresholdValues", [y, scores], np.asarray(c[4])),
        ]
        for drop in [True, False]:
            fpr, tpr, thr = M.roc_curve(y, scores, drop_intermediate=drop)
            out += [
                case(f"rocCurve/{label}/drop={drop}/fpr", "rocFpr", [y, scores], fpr, dropIntermediate=drop),
                case(f"rocCurve/{label}/drop={drop}/tpr", "rocTpr", [y, scores], tpr, dropIntermediate=drop),
                case(
                    f"rocCurve/{label}/drop={drop}/thresholds", "rocThresholds", [y, scores], thr, dropIntermediate=drop
                ),
            ]
        fpr, fnr, thr = M.det_curve(y, scores)
        out += [
            case(f"detCurve/{label}/fpr", "detFpr", [y, scores], fpr),
            case(f"detCurve/{label}/fnr", "detFnr", [y, scores], fnr),
            case(f"detCurve/{label}/thresholds", "detThresholds", [y, scores], thr),
        ]
        out.append(case(f"rocAucScore/{label}", "rocAucScore", [y, scores], np.asarray(M.roc_auc_score(y, scores))))
    fpr, tpr, thr = M.roc_curve(y, s, sample_weight=w)
    out += [
        case("rocCurve/weighted/fpr", "rocFpr", [y, s, w], fpr),
        case("rocCurve/weighted/tpr", "rocTpr", [y, s, w], tpr),
        case("rocAucScore/weighted", "rocAucScore", [y, s, w], np.asarray(M.roc_auc_score(y, s, sample_weight=w))),
        case("rocAucScore/maxFpr", "rocAucScore", [y, s], np.asarray(M.roc_auc_score(y, s, max_fpr=0.3)), maxFpr=0.3),
        case(
            "rocAucScore/maxFpr-weighted",
            "rocAucScore",
            [y, s, w],
            np.asarray(M.roc_auc_score(y, s, max_fpr=0.1, sample_weight=w)),
            maxFpr=0.1,
        ),
        case(
            "rocAucScore/ovr-macro",
            "rocAucScore",
            [yk, probs],
            np.asarray(M.roc_auc_score(yk, probs, multi_class="ovr")),
        ),
        case(
            "rocAucScore/ovr-weighted",
            "rocAucScore",
            [yk, probs, w],
            np.asarray(M.roc_auc_score(yk, probs, multi_class="ovr", average="weighted", sample_weight=w)),
            average="weighted",
            multiClass="ovr",
        ),
        case(
            "rocAucScore/ovo-macro",
            "rocAucScore",
            [yk, probs],
            np.asarray(M.roc_auc_score(yk, probs, multi_class="ovo")),
            multiClass="ovo",
        ),
        case(
            "rocAucScore/ovo-weighted",
            "rocAucScore",
            [yk, probs],
            np.asarray(M.roc_auc_score(yk, probs, multi_class="ovo", average="weighted")),
            multiClass="ovo",
            average="weighted",
        ),
        case("rocAucScore/multilabel-macro", "rocAucScore", [ym, sm], np.asarray(M.roc_auc_score(ym, sm))),
        case(
            "rocAucScore/multilabel-micro",
            "rocAucScore",
            [ym, sm, w],
            np.asarray(M.roc_auc_score(ym, sm, average="micro", sample_weight=w)),
            average="micro",
        ),
        case(
            "rocAucScore/multilabel-weighted",
            "rocAucScore",
            [ym, sm],
            np.asarray(M.roc_auc_score(ym, sm, average="weighted")),
            average="weighted",
        ),
        case(
            "rocAucScore/two-columns",
            "rocAucScore",
            [y, np.stack([1 - s, s], axis=1)],
            np.asarray(M.roc_auc_score(y, s)),
        ),
        case("auc", "auc", [fpr, tpr], np.asarray(M.auc(fpr, tpr))),
        case("auc/decreasing", "auc", [fpr[::-1], tpr[::-1]], np.asarray(M.auc(fpr[::-1], tpr[::-1]))),
    ]
    return out
