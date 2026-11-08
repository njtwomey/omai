"""External clustering metrics against sklearn.metrics.cluster; purity and B-cubed from their definitions."""

import numpy as np
import sklearn
import sklearn.metrics as M
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def purity(t: FloatArray, p: FloatArray) -> float:
    c = np.asarray(contingency_matrix(t, p))
    return float(c.max(axis=0).sum() / c.sum())


def bcubed(t: FloatArray, p: FloatArray) -> tuple[float, float, float]:
    same_class = t[:, None] == t[None, :]
    same_cluster = p[:, None] == p[None, :]
    both = (same_class & same_cluster).sum(1)
    precision = float((both / same_cluster.sum(1)).mean())
    recall = float((both / same_class.sum(1)).mean())
    return precision, recall, 2 * precision * recall / (precision + recall)


def cases() -> list[Case]:
    g = rng(61)
    n = 40
    t = g.integers(0, 4, size=n).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.6, t, g.integers(0, 5, size=n)).astype(np.float64)
    out: list[Case] = [
        case("contingencyMatrix", "contingencyMatrix", [t, p], np.asarray(contingency_matrix(t, p), dtype=np.float64)),
        case(
            "pairConfusionMatrix",
            "pairConfusionMatrix",
            [t, p],
            np.asarray(pair_confusion_matrix(t, p), dtype=np.float64),
        ),
        case("randScore", "randScore", [t, p], np.asarray(M.rand_score(t, p))),
        case("adjustedRandScore", "adjustedRandScore", [t, p], np.asarray(M.adjusted_rand_score(t, p))),
        case("fowlkesMallowsScore", "fowlkesMallowsScore", [t, p], np.asarray(M.fowlkes_mallows_score(t, p))),
        case("mutualInfoScore", "mutualInfoScore", [t, p], np.asarray(M.mutual_info_score(t, p))),
        case("homogeneityScore", "homogeneityScore", [t, p], np.asarray(M.homogeneity_score(t, p))),
        case("completenessScore", "completenessScore", [t, p], np.asarray(M.completeness_score(t, p))),
        case("vMeasureScore", "vMeasureScore", [t, p], np.asarray(M.v_measure_score(t, p))),
        case("vMeasureScore/beta", "vMeasureScore", [t, p], np.asarray(M.v_measure_score(t, p, beta=2.0)), beta=2.0),
        case("purityScore", "purityScore", [t, p], np.asarray(purity(t, p))),
        case("bCubedF1", "bCubedF1", [t, p], np.asarray(bcubed(t, p)[2])),
        case("bCubedPrecision", "bCubedPrecision", [t, p], np.asarray(bcubed(t, p)[0])),
        case("bCubedRecall", "bCubedRecall", [t, p], np.asarray(bcubed(t, p)[1])),
        case(
            "adjustedRandScore/single",
            "adjustedRandScore",
            [t, np.zeros(n)],
            np.asarray(M.adjusted_rand_score(t, np.zeros(n))),
        ),
        case(
            "normalizedMutualInfoScore/single",
            "normalizedMutualInfoScore",
            [np.zeros(n), np.zeros(n)],
            np.asarray(M.normalized_mutual_info_score(np.zeros(n), np.zeros(n))),
        ),
    ]
    for method in ["arithmetic", "geometric", "min", "max"]:
        out += [
            case(
                f"normalizedMutualInfoScore/{method}",
                "normalizedMutualInfoScore",
                [t, p],
                np.asarray(M.normalized_mutual_info_score(t, p, average_method=method)),
                averageMethod=method,
            ),
            case(
                f"adjustedMutualInfoScore/{method}",
                "adjustedMutualInfoScore",
                [t, p],
                np.asarray(M.adjusted_mutual_info_score(t, p, average_method=method)),
                averageMethod=method,
                rtol=1e-10,
            ),
        ]
    return out
