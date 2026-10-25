"""Agreement, likelihood-ratio, multilabel-ranking and report metrics against sklearn.metrics."""

from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# sklearn types zero_division as a string; the keyword passes through a mapping instead.
ZERO: dict[str, Any] = {"zero_division": 0}


def cases() -> list[Case]:
    g = rng(52)
    n = 40
    t = g.integers(0, 4, size=n).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.6, t, g.integers(0, 4, size=n)).astype(np.float64)
    w = g.uniform(0.2, 2.0, size=n)
    tb = (g.uniform(size=n) > 0.4).astype(np.float64)
    pb = np.where(g.uniform(size=n) < 0.7, tb, 1 - tb)
    tm = (g.uniform(size=(n, 4)) > 0.6).astype(np.float64)
    pm = np.where(g.uniform(size=(n, 4)) < 0.75, tm, 1 - tm)
    scores = g.normal(size=(n, 4))
    scores[:5, 1] = scores[:5, 2]  # ties
    probs = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
    lr_pos, lr_neg = M.class_likelihood_ratios(tb, pb)
    lr_pos_w, lr_neg_w = M.class_likelihood_ratios(tb, pb, sample_weight=w)
    report_true = [0, 1, 2, 2, 2, 1, 0, 3]
    report_pred = [0, 0, 2, 2, 1, 1, 0, 2]
    out: list[Case] = [
        case("matthewsCorrcoef", "matthewsCorrcoef", [t, p], np.asarray(M.matthews_corrcoef(t, p))),
        case(
            "matthewsCorrcoef/binary-weighted",
            "matthewsCorrcoef",
            [tb, pb, w],
            np.asarray(M.matthews_corrcoef(tb, pb, sample_weight=w)),
        ),
        case("cohenKappaScore", "cohenKappaScore", [t, p], np.asarray(M.cohen_kappa_score(t, p))),
        case(
            "cohenKappaScore/linear",
            "cohenKappaScore",
            [t, p],
            np.asarray(M.cohen_kappa_score(t, p, weights="linear")),
            weights="linear",
        ),
        case(
            "cohenKappaScore/quadratic-weighted",
            "cohenKappaScore",
            [t, p, w],
            np.asarray(M.cohen_kappa_score(t, p, weights="quadratic", sample_weight=w)),
            weights="quadratic",
        ),
        case(
            "quadraticWeightedKappa",
            "quadraticWeightedKappa",
            [t, p],
            np.asarray(M.cohen_kappa_score(t, p, weights="quadratic")),
        ),
        case("positiveLikelihoodRatio", "positiveLikelihoodRatio", [tb, pb], np.asarray(lr_pos)),
        case("negativeLikelihoodRatio", "negativeLikelihoodRatio", [tb, pb], np.asarray(lr_neg)),
        case("positiveLikelihoodRatio/weighted", "positiveLikelihoodRatio", [tb, pb, w], np.asarray(lr_pos_w)),
        case("negativeLikelihoodRatio/weighted", "negativeLikelihoodRatio", [tb, pb, w], np.asarray(lr_neg_w)),
        case("diagnosticOddsRatio", "diagnosticOddsRatio", [tb, pb], np.asarray(lr_pos / lr_neg)),
        case("hammingLoss", "hammingLoss", [t, p], np.asarray(M.hamming_loss(t, p))),
        case(
            "hammingLoss/multilabel-weighted",
            "hammingLoss",
            [tm, pm, w],
            np.asarray(M.hamming_loss(tm, pm, sample_weight=w)),
        ),
        case("zeroOneLoss", "zeroOneLoss", [t, p], np.asarray(M.zero_one_loss(t, p))),
        case(
            "zeroOneLoss/count-weighted",
            "zeroOneLoss",
            [t, p, w],
            np.asarray(M.zero_one_loss(t, p, normalize=False, sample_weight=w)),
            normalize=False,
        ),
        case("zeroOneLoss/multilabel", "zeroOneLoss", [tm, pm], np.asarray(M.zero_one_loss(tm, pm))),
    ]
    for k in [1, 2, 3]:
        out.append(
            case(
                f"topKAccuracyScore/k={k}",
                "topKAccuracyScore",
                [t, probs],
                np.asarray(M.top_k_accuracy_score(t, probs, k=k, labels=[0, 1, 2, 3])),
                k=k,
            )
        )
    out += [
        case(
            "topKAccuracyScore/weighted",
            "topKAccuracyScore",
            [t, scores, w],
            np.asarray(M.top_k_accuracy_score(t, scores, k=2, sample_weight=w, labels=[0, 1, 2, 3])),
        ),
        case(
            "topKAccuracyScore/binary",
            "topKAccuracyScore",
            [tb, probs[:, 0]],
            np.asarray(M.top_k_accuracy_score(tb, probs[:, 0], k=1)),
            k=1,
        ),
        case(
            "topKAccuracyScore/binary-decision",
            "topKAccuracyScore",
            [tb, scores[:, 0]],
            np.asarray(M.top_k_accuracy_score(tb, scores[:, 0], k=1)),
            k=1,
        ),
        case(
            "labelRankingAveragePrecisionScore",
            "labelRankingAveragePrecisionScore",
            [tm, scores],
            np.asarray(M.label_ranking_average_precision_score(tm, scores)),
        ),
        case(
            "labelRankingAveragePrecisionScore/weighted",
            "labelRankingAveragePrecisionScore",
            [tm, scores, w],
            np.asarray(M.label_ranking_average_precision_score(tm, scores, sample_weight=w)),
        ),
        case("coverageError", "coverageError", [tm, scores], np.asarray(M.coverage_error(tm, scores))),
        case(
            "coverageError/weighted",
            "coverageError",
            [tm, scores, w],
            np.asarray(M.coverage_error(tm, scores, sample_weight=w)),
        ),
        case("labelRankingLoss", "labelRankingLoss", [tm, scores], np.asarray(M.label_ranking_loss(tm, scores))),
        case(
            "labelRankingLoss/weighted",
            "labelRankingLoss",
            [tm, scores, w],
            np.asarray(M.label_ranking_loss(tm, scores, sample_weight=w)),
        ),
    ]
    reports: list[Case] = [
        case(
            "classificationReport",
            "classificationReport",
            [],
            str(M.classification_report(report_true, report_pred, **ZERO)),
            yTrue=report_true,
            yPred=report_pred,
        ),
        case(
            "classificationReport/digits-names",
            "classificationReport",
            [],
            str(
                M.classification_report(
                    report_true, report_pred, digits=3, **ZERO, target_names=["zero", "one", "two", "three"]
                )
            ),
            yTrue=report_true,
            yPred=report_pred,
            digits=3,
            targetNames=["zero", "one", "two", "three"],
        ),
        case(
            "classificationReport/labels",
            "classificationReport",
            [],
            str(M.classification_report(report_true, report_pred, labels=[0, 2], **ZERO)),
            yTrue=report_true,
            yPred=report_pred,
            labels=[0, 2],
        ),
        case(
            "classificationReport/multilabel",
            "classificationReport",
            [],
            str(M.classification_report(tm[:10].astype(int), pm[:10].astype(int), **ZERO)),
            yTrue=tm[:10].astype(int).tolist(),
            yPred=pm[:10].astype(int).tolist(),
        ),
        case(
            "classificationReport/weighted",
            "classificationReport",
            [],
            str(M.classification_report(report_true, report_pred, **ZERO, sample_weight=[1, 2, 1, 1, 0.5, 1, 1, 2])),
            yTrue=report_true,
            yPred=report_pred,
            sampleWeight=[1, 2, 1, 1, 0.5, 1, 1, 2],
        ),
    ]
    return out + reports
