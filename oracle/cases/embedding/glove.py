"""The GloVe objective and its gradient against numpy: the harmonic-weighted symmetric co-occurrence matrix of a few
sentences (the definition of Pennington et al. 2014), the weight function min((x / xmax)^alpha, 1) and the weighted
least-squares loss (with its factor one half, as in the reference C code) at random parameters, averaged over the
nonzero entries. The parameters are the flat vector [W, W~, b, b~]."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

SENTENCES: list[list[str]] = [
    s.split()
    for s in [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat saw a dog in the park",
        "the dog saw the cat",
        "a bird sat on the log near the mat",
    ]
]


def cooccurrence(window: int) -> tuple[list[str], FloatArray]:
    words = sorted({w for s in SENTENCES for w in s})
    idx = {w: i for i, w in enumerate(words)}
    x = np.zeros((len(words), len(words)))
    for s in SENTENCES:
        for i, w in enumerate(s):
            for d in range(1, window + 1):
                for j in (i - d, i + d):
                    if 0 <= j < len(s):
                        x[idx[w], idx[s[j]]] += 1.0 / d
    return words, x


def loss_and_grad(x: FloatArray, theta: FloatArray, dim: int, xmax: float, alpha: float) -> tuple[float, FloatArray]:
    v = x.shape[0]
    w = theta[: v * dim].reshape(v, dim)
    wt = theta[v * dim : 2 * v * dim].reshape(v, dim)
    b = theta[2 * v * dim : 2 * v * dim + v]
    bt = theta[2 * v * dim + v :]
    i, j = np.nonzero(x)
    counts = x[i, j]
    f = np.minimum((counts / xmax) ** alpha, 1.0)
    diff = np.sum(w[i] * wt[j], axis=1) + b[i] + bt[j] - np.log(counts)
    n = len(counts)
    loss = float(np.sum(0.5 * f * diff**2) / n)
    fd = f * diff / n
    gw = np.zeros_like(w)
    gwt = np.zeros_like(wt)
    gb = np.zeros(v)
    gbt = np.zeros(v)
    np.add.at(gw, i, fd[:, None] * wt[j])
    np.add.at(gwt, j, fd[:, None] * w[i])
    np.add.at(gb, i, fd)
    np.add.at(gbt, j, fd)
    return loss, np.concatenate([gw.ravel(), gwt.ravel(), gb, gbt])


def cases() -> list[Case]:
    out: list[Case] = []
    for window, dim, xmax, alpha, seed in [(2, 3, 100.0, 0.75, 1), (3, 4, 2.0, 0.75, 2), (4, 2, 1.0, 0.5, 3)]:
        words, x = cooccurrence(window)
        v = len(words)
        theta = rng(seed).normal(scale=0.5, size=2 * v * dim + 2 * v)
        loss, grad = loss_and_grad(x, theta, dim, xmax, alpha)
        out.append(
            case(
                f"loss/{window}/{dim}/{xmax}/{alpha}",
                "gloveLoss",
                [SENTENCES, theta],
                np.concatenate([[loss], grad]),
                rtol=1e-10,
                atol=1e-12,
                window=window,
                dim=dim,
                xmax=xmax,
                alpha=alpha,
                words=words,
            )
        )
    return out
