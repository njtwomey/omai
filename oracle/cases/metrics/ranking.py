"""Ranking metrics against sklearn (DCG, NDCG) and torchmetrics' retrieval metrics, with trec_eval's average precision,
expected reciprocal rank, rank-biased precision and the beyond-accuracy measures from their definitions.
"""

from collections.abc import Callable
from importlib.metadata import version

import numpy as np
import sklearn
import sklearn.metrics as M
import torch
import torchmetrics.functional.retrieval as R
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_torch

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "torchmetrics": version("torchmetrics")}

# torchmetrics computes retrieval metrics in float32.
F32 = 1e-6


def per_query(rel: FloatArray, scores: FloatArray, f: Callable[..., torch.Tensor], **kw: int) -> float:
    """The mean over queries of a torchmetrics retrieval function."""
    values = [float(f(to_torch(scores[q]), torch.tensor(rel[q] > 0), **kw)) for q in range(len(rel))]
    return float(np.mean(values))


def ranked(rel: FloatArray, scores: FloatArray) -> FloatArray:
    return rel[np.argsort(-scores, kind="stable")]


def trec_ap(rel: FloatArray, scores: FloatArray, k: int | None) -> float:
    out: list[float] = []
    for q in range(len(rel)):
        r = ranked(rel[q], scores[q]) > 0
        top = r[: k or len(r)]
        hits = np.cumsum(top)
        precisions = hits[top] / (np.flatnonzero(top) + 1)
        total = r.sum()
        out.append(float(precisions.sum() / total) if total else 0.0)
    return float(np.mean(out))


def err(rel: FloatArray, scores: FloatArray) -> float:
    gmax = rel.max()
    out: list[float] = []
    for q in range(len(rel)):
        p = (2.0 ** ranked(rel[q], scores[q]) - 1) / 2.0**gmax
        still = np.concatenate([[1.0], np.cumprod(1 - p)[:-1]])
        out.append(float((still * p / np.arange(1, len(p) + 1)).sum()))
    return float(np.mean(out))


def rbp(rel: FloatArray, scores: FloatArray, p: float) -> float:
    return float(
        np.mean(
            [(1 - p) * ((ranked(rel[q], scores[q]) > 0) * p ** np.arange(rel.shape[1])).sum() for q in range(len(rel))]
        )
    )


def ndcg_exponential(rel: FloatArray, scores: FloatArray) -> float:
    return float(M.ndcg_score(2.0**rel - 1, scores))


def cases() -> list[Case]:
    g = rng(60)
    q, items = 6, 8
    rel = g.integers(0, 4, size=(q, items)).astype(np.float64)
    rel[0] = 0
    scores = 1 / (1 + np.exp(-(0.6 * rel + g.normal(size=(q, items)))))
    tied = np.round(scores, 1)
    w = g.uniform(0.5, 2.0, size=q)
    binary = (rel > 1).astype(np.float64)
    out: list[Case] = [
        case("dcgScore", "dcgScore", [rel, scores], np.asarray(M.dcg_score(rel, scores))),
        case("dcgScore/k3-tied", "dcgScore", [rel, tied], np.asarray(M.dcg_score(rel, tied, k=3)), k=3),
        case(
            "dcgScore/ignore-ties",
            "dcgScore",
            [rel, tied],
            np.asarray(M.dcg_score(rel, tied, ignore_ties=True)),
            ignoreTies=True,
        ),
        case(
            "dcgScore/base10-weighted",
            "dcgScore",
            [rel, scores, w],
            np.asarray(M.dcg_score(rel, scores, log_base=10, sample_weight=w)),
            logBase=10,
        ),
        case("ndcgScore", "ndcgScore", [rel, scores], np.asarray(M.ndcg_score(rel, scores))),
        case("ndcgScore/k4-tied", "ndcgScore", [rel, tied], np.asarray(M.ndcg_score(rel, tied, k=4)), k=4),
        case(
            "ndcgScore/weighted", "ndcgScore", [rel, scores, w], np.asarray(M.ndcg_score(rel, scores, sample_weight=w))
        ),
        case(
            "ndcgScore/exponential",
            "ndcgScore",
            [rel, scores],
            np.asarray(ndcg_exponential(rel, scores)),
            gain="exponential",
        ),
    ]
    for k in [3, 5]:
        out += [
            case(
                f"precisionAtK/{k}",
                "precisionAtK",
                [binary, scores],
                np.asarray(per_query(binary, scores, R.retrieval_precision, top_k=k)),
                k=k,
                rtol=F32,
            ),
            case(
                f"recallAtK/{k}",
                "recallAtK",
                [binary, scores],
                np.asarray(per_query(binary, scores, R.retrieval_recall, top_k=k)),
                k=k,
                rtol=F32,
            ),
            case(
                f"hitRate/{k}",
                "hitRate",
                [binary, scores],
                np.asarray(per_query(binary, scores, R.retrieval_hit_rate, top_k=k)),
                k=k,
                rtol=F32,
            ),
            case(
                f"meanReciprocalRank/{k}",
                "meanReciprocalRank",
                [binary, scores],
                np.asarray(per_query(binary, scores, R.retrieval_reciprocal_rank, top_k=k)),
                k=k,
                rtol=F32,
            ),
            case(
                f"meanAveragePrecision/retrieved-{k}",
                "meanAveragePrecision",
                [binary, scores],
                np.asarray(per_query(binary, scores, R.retrieval_average_precision, top_k=k)),
                k=k,
                denominator="retrieved",
                rtol=F32,
            ),
            case(
                f"meanAveragePrecision/relevant-{k}",
                "meanAveragePrecision",
                [binary, scores],
                np.asarray(trec_ap(binary, scores, k)),
                k=k,
            ),
        ]
    out += [
        case(
            "rPrecision",
            "rPrecision",
            [binary, scores],
            np.asarray(per_query(binary, scores, R.retrieval_r_precision)),
            rtol=F32,
        ),
        case(
            "meanAveragePrecision", "meanAveragePrecision", [binary, scores], np.asarray(trec_ap(binary, scores, None))
        ),
        case(
            "ndcgScore/torchmetrics",
            "ndcgScore",
            [rel, scores],
            np.asarray(
                np.mean(
                    [
                        float(R.retrieval_normalized_dcg(to_torch(scores[i]), torch.tensor(rel[i]).long()))
                        for i in range(q)
                    ]
                )
            ),
            rtol=F32,
        ),
        case("expectedReciprocalRank", "expectedReciprocalRank", [rel, scores], np.asarray(err(rel, scores))),
        case("rankBiasedPrecision", "rankBiasedPrecision", [binary, scores], np.asarray(rbp(binary, scores, 0.8))),
        case(
            "rankBiasedPrecision/p=0.5",
            "rankBiasedPrecision",
            [binary, scores],
            np.asarray(rbp(binary, scores, 0.5)),
            persistence=0.5,
        ),
    ]
    return out
