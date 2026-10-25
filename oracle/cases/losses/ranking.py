"""Learning-to-rank losses against torch.nn.functional and optax, and their defining formulas in numpy (float64).

RankNet, LambdaRank, ListNet, ListMLE and ApproxNDCG have no function in torch, optax or scikit-learn; their references
are the published formulas written directly over the lists.
"""

import jax
import numpy as np
import optax
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {"torch": torch.__version__, "optax": optax.__version__}


def softplus(x: FloatArray) -> FloatArray:
    return np.logaddexp(0.0, x)


def preferred_pairs(rel: FloatArray) -> list[tuple[int, int]]:
    """The pairs (i, j) of one list with rel_i > rel_j."""
    n = len(rel)
    return [(i, j) for i in range(n) for j in range(n) if rel[i] > rel[j]]


def ranknet(s: FloatArray, r: FloatArray, sigma: float) -> float:
    terms = [
        softplus(np.asarray(-sigma * (s[b, i] - s[b, j]))) for b in range(len(s)) for i, j in preferred_pairs(r[b])
    ]
    return float(np.mean(terms))


def pairwise_hinge(s: FloatArray, r: FloatArray, margin: float) -> float:
    terms = [max(0.0, margin - (s[b, i] - s[b, j])) for b in range(len(s)) for i, j in preferred_pairs(r[b])]
    return float(np.mean(terms))


def dcg_discount(rank: int, k: int | None) -> float:
    return 0.0 if k is not None and rank > k else 1.0 / np.log2(1 + rank)


def lambdarank(s: FloatArray, r: FloatArray, sigma: float, k: int | None) -> float:
    per_list: list[float] = []
    for b in range(len(s)):
        gain = 2.0 ** r[b] - 1
        order = np.argsort(-s[b], kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(1, len(order) + 1)
        ideal = sum(g * dcg_discount(i + 1, k) for i, g in enumerate(np.sort(gain)[::-1]))
        total = 0.0
        for i, j in preferred_pairs(r[b]):
            delta = abs(gain[i] - gain[j]) * abs(dcg_discount(int(rank[i]), k) - dcg_discount(int(rank[j]), k)) / ideal
            total += delta * float(softplus(np.asarray(-sigma * (s[b, i] - s[b, j]))))
        per_list.append(total)
    return float(np.mean(per_list))


def log_softmax(x: FloatArray) -> FloatArray:
    return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)


def listnet(s: FloatArray, r: FloatArray, temperature: float) -> FloatArray:
    target = np.exp(log_softmax(r / temperature))
    return -(target * log_softmax(s)).sum(-1)


def listmle(s: FloatArray, r: FloatArray) -> FloatArray:
    out: list[float] = []
    for b in range(len(s)):
        ordered = s[b][np.argsort(-r[b], kind="stable")]
        out.append(sum(np.logaddexp.reduce(ordered[i:]) - ordered[i] for i in range(len(ordered))))
    return np.asarray(out)


def approx_ndcg(s: FloatArray, r: FloatArray, alpha: float) -> FloatArray:
    out: list[float] = []
    for b in range(len(s)):
        n = len(s[b])
        ranks = [1 + sum(1 / (1 + np.exp(-alpha * (s[b, j] - s[b, i]))) for j in range(n) if j != i) for i in range(n)]
        gain = 2.0 ** r[b] - 1
        dcg = sum(gain[i] / np.log2(1 + ranks[i]) for i in range(n))
        ideal = sum(g / np.log2(2 + i) for i, g in enumerate(np.sort(gain)[::-1]))
        out.append(-dcg / ideal if ideal > 0 else 0.0)
    return np.asarray(out)


def cases() -> list[Case]:
    g = rng(44)
    s = g.normal(size=(3, 6))
    r = g.integers(0, 4, size=(3, 6)).astype(np.float64)
    x1, x2 = g.normal(size=8), g.normal(size=8)
    y = np.where(g.uniform(size=8) > 0.5, 1.0, -1.0)
    pos, negs = g.normal(size=(5, 1)), g.normal(size=(5, 4))
    return [
        case(
            "marginRanking",
            "marginRanking",
            [x1, x2, y],
            to_numpy(F.margin_ranking_loss(to_torch(x1), to_torch(x2), to_torch(y), margin=0.3)),
            margin=0.3,
        ),
        case("bpr", "bpr", [pos, negs], np.asarray(softplus(negs - pos).mean())),
        case("rankNet", "rankNet", [s, r], np.asarray(ranknet(s, r, 1.0))),
        case("rankNet/sigma", "rankNet", [s, r], np.asarray(ranknet(s, r, 2.5)), sigma=2.5),
        case("pairwiseHinge", "pairwiseHinge", [s, r], np.asarray(pairwise_hinge(s, r, 1.0))),
        case("lambdaRank", "lambdaRank", [s, r], np.asarray(lambdarank(s, r, 1.0, None))),
        case("lambdaRank/k=3", "lambdaRank", [s, r], np.asarray(lambdarank(s, r, 1.5, 3)), sigma=1.5, k=3),
        case("listNet", "listNet", [s, r], listnet(s, r, 1.0), reduction="none"),
        case("listNet/temperature", "listNet", [s, r], np.asarray(listnet(s, r, 0.5).mean()), temperature=0.5),
        case(
            "softmaxRanking",
            "softmaxRanking",
            [s, r],
            np.asarray(optax.losses.ranking_softmax_loss(s, r, reduce_fn=None), dtype=np.float64),
            reduction="none",
        ),
        case("listMle", "listMle", [s, r], listmle(s, r), reduction="none"),
        case("approxNdcg", "approxNdcg", [s, r], approx_ndcg(s, r, 10.0), reduction="none"),
        case("approxNdcg/alpha", "approxNdcg", [s, r], np.asarray(approx_ndcg(s, r, 2.0).mean()), alpha=2.0),
    ]
