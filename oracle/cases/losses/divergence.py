"""Divergence losses against torch.nn.functional, scipy, scikit-learn kernels and POT (float64)."""

from typing import Any

import numpy as np
import ot
import scipy
import scipy.spatial.distance as ssd
import scipy.stats as st
import sklearn
import torch
import torch.nn.functional as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

REFERENCE: dict[str, str] = {
    "torch": torch.__version__,
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "pot": ot.__version__,
}


def mmd2(x: FloatArray, y: FloatArray, kernel: str, bandwidths: list[float], unbiased: bool) -> float:
    def k(a: FloatArray, b: FloatArray) -> FloatArray:
        if kernel == "energy":
            return -euclidean_distances(a, b)
        if kernel == "gaussian":
            return np.sum([rbf_kernel(a, b, gamma=1 / (2 * s * s)) for s in bandwidths], axis=0)
        return np.sum([np.exp(-euclidean_distances(a, b) / s) for s in bandwidths], axis=0)

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    if unbiased:
        n, m = len(x), len(y)
        return float(
            (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        )
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def median_bandwidth(x: FloatArray, y: FloatArray) -> float:
    return float(np.median(ssd.pdist(np.concatenate([x, y]))))


def cases() -> list[Case]:
    g = rng(48)
    logits_p, logits_q = g.normal(size=(4, 5)), g.normal(size=(4, 5))
    p = np.exp(logits_p) / np.exp(logits_p).sum(-1, keepdims=True)
    q = np.exp(logits_q) / np.exp(logits_q).sum(-1, keepdims=True)
    lq = np.log(q)
    labels = np.array([0.0, 3, 1, 4])
    out: list[Case] = []
    for reduction in ["mean", "sum", "none", "batchmean"]:
        r: dict[str, Any] = {"reduction": reduction}
        out.append(
            case(f"klDiv/{reduction}", "klDiv", [lq, p], to_numpy(F.kl_div(to_torch(lq), to_torch(p), **r)), **r)
        )
    out += [
        case(
            "klDiv/logTarget",
            "klDiv",
            [lq, np.log(p)],
            to_numpy(F.kl_div(to_torch(lq), to_torch(np.log(p)), reduction="batchmean", log_target=True)),
            reduction="batchmean",
            logTarget=True,
        ),
        case("klDivergence", "klDivergence", [p, q], st.entropy(p, q, axis=-1), reduction="none"),
        case(
            "klDivergence/logits",
            "klDivergence",
            [logits_p, logits_q],
            np.asarray(st.entropy(p, q, axis=-1).mean()),
            logits=True,
        ),
        case(
            "jensenShannon",
            "jensenShannon",
            [p, q],
            np.asarray([ssd.jensenshannon(a, b) ** 2 for a, b in zip(p, q, strict=True)]),
            reduction="none",
        ),
        case(
            "jensenShannon/base2-logits",
            "jensenShannon",
            [logits_p, logits_q],
            np.asarray(np.mean([ssd.jensenshannon(a, b, base=2) ** 2 for a, b in zip(p, q, strict=True)])),
            logits=True,
            base=2,
        ),
    ]
    tp, tq = to_torch(logits_p), to_torch(logits_q)
    temp = 2.5
    soft = F.kl_div(F.log_softmax(tp / temp, -1), F.log_softmax(tq / temp, -1), reduction="batchmean", log_target=True)
    hard = F.cross_entropy(tp, torch.tensor(labels, dtype=torch.long))
    out += [
        case("distillation", "distillation", [logits_p, logits_q], to_numpy(soft * temp**2), temperature=temp),
        case(
            "distillation/hard",
            "distillation",
            [logits_p, logits_q, labels],
            to_numpy(0.7 * soft * temp**2 + 0.3 * hard),
            temperature=temp,
            alpha=0.7,
        ),
    ]
    x, y = g.normal(size=(6, 2)), g.normal(size=(5, 2)) + 0.5
    out += [
        case("mmd/gaussian", "mmd", [x, y], np.asarray(mmd2(x, y, "gaussian", [1.0], False)), bandwidths=[1.0]),
        case(
            "mmd/multiscale-unbiased",
            "mmd",
            [x, y],
            np.asarray(mmd2(x, y, "gaussian", [0.5, 1.0, 2.0], True)),
            bandwidths=[0.5, 1.0, 2.0],
            unbiased=True,
        ),
        case(
            "mmd/median",
            "mmd",
            [x, y],
            np.asarray(mmd2(x, y, "gaussian", [median_bandwidth(x, y)], False)),
            rtol=1e-10,
        ),
        case(
            "mmd/laplacian",
            "mmd",
            [x, y],
            np.asarray(mmd2(x, y, "laplacian", [0.8], False)),
            kernel="laplacian",
            bandwidths=[0.8],
        ),
        case("mmd/energy", "mmd", [x, y], np.asarray(mmd2(x, y, "energy", [1.0], False)), kernel="energy"),
    ]
    a = g.uniform(0.5, 1.5, size=6)
    a /= a.sum()
    b = np.full(5, 0.2)
    cost = ot.dist(x, y)
    out += [
        case(
            "sinkhorn",
            "sinkhorn",
            [x, y],
            np.asarray(
                ot.sinkhorn2(np.full(6, 1 / 6), b, cost, 0.5, method="sinkhorn_log", stopThr=1e-13, numItermax=100000)
            ),
            epsilon=0.5,
            tol=1e-13,
            maxIter=100000,
            rtol=1e-9,
            atol=1e-12,
        ),
        case(
            "sinkhorn/weighted",
            "sinkhorn",
            [x, y, a, b],
            np.asarray(ot.sinkhorn2(a, b, cost, 0.3, method="sinkhorn_log", stopThr=1e-13, numItermax=100000)),
            epsilon=0.3,
            tol=1e-13,
            maxIter=100000,
            rtol=1e-9,
            atol=1e-12,
        ),
        case(
            "sinkhorn/debiased",
            "sinkhorn",
            [x, y],
            np.asarray(
                ot.bregman.empirical_sinkhorn_divergence(
                    x, y, 0.5, method="sinkhorn_log", stopThr=1e-13, numIterMax=100000
                )
            ),
            epsilon=0.5,
            debiased=True,
            tol=1e-13,
            maxIter=100000,
            rtol=1e-9,
            atol=1e-12,
        ),
    ]
    return out
