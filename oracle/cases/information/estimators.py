"""Estimators from samples against scipy, sklearn, NPEET (k-NN estimators) and pyitlib (plug-in transfer entropy)."""

import importlib.metadata
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy
import scipy.special as sp
import scipy.stats as st
import sklearn
import sklearn.metrics as skm
from npeet import entropy_estimators as ee  # pyright: ignore[reportMissingTypeStubs]
from pyitlib import discrete_random_variable as drv  # pyright: ignore[reportMissingTypeStubs]
from sklearn.feature_selection._mutual_info import (  # pyright: ignore[reportPrivateUsage]
    _compute_mi_cc,  # pyright: ignore[reportPrivateUsage]
    _compute_mi_cd,  # pyright: ignore[reportPrivateUsage]
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

# NPEET and pyitlib annotate `base` as int through their defaults; any positive base works, and e gives nats.
NATS: Any = np.e

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "npeet": importlib.metadata.version("npeet"),
    "pyitlib": importlib.metadata.version("pyitlib"),
}

# NPEET adds noise of size 1e-10 to break ties; the estimates move by about that much, far inside these tolerances.
KNN_RTOL = 1e-7


def seeded(f: Callable[..., object], *args: object, **kwargs: object) -> float:
    """Call an NPEET estimator with numpy's global generator seeded (its jitter uses np.random)."""
    np.random.seed(0)  # noqa: NPY002
    return float(cast(float, f(*args, **kwargs)))


def kde_entropy(x: FloatArray) -> float:
    """-mean log f(x_i) for scipy's Gaussian KDE (Scott's rule) fitted to x of shape [n, d]."""
    kde = st.gaussian_kde(x.T)
    return float(-np.mean(kde.logpdf(x.T)))


def cases() -> list[Case]:
    g = rng(420)
    out: list[Case] = []
    # Discrete samples.
    a = g.integers(0, 5, size=60).astype(np.float64)
    b = (a + g.integers(0, 3, size=60)) % 4
    counts = np.unique(a, return_counts=True)[1]
    h = float(st.entropy(counts))
    out.append(case("plugInEntropy", "plugInEntropy", [a], np.array(h)))
    out.append(case("plugInEntropy/base=2", "plugInEntropy", [a], np.array(h / np.log(2)), base=2))
    mm = h + (len(counts) - 1) / (2 * len(a))
    out.append(case("plugInEntropy/millerMadow", "plugInEntropy", [a], np.array(mm), correction="millerMadow"))
    mi = skm.mutual_info_score(a, b)
    out.append(case("plugInMutualInformation", "plugInMutualInformation", [a, b], np.array(mi)))

    # Continuous samples for the k-nearest-neighbour estimators.
    n = 300
    x1 = g.normal(size=(n, 1))
    x2 = g.normal(size=(n, 2))
    y1 = x1 + 0.5 * g.normal(size=(n, 1))
    y2 = np.c_[x2[:, :1] ** 2, x2[:, 1:] - x2[:, :1]] + 0.3 * g.normal(size=(n, 2))
    z1 = x1 + g.normal(size=(n, 1))
    for k in [3, 5]:
        out.append(
            case(
                f"kozachenkoLeonenkoEntropy/1d/k={k}",
                "kozachenkoLeonenkoEntropy",
                [x1],
                np.array(seeded(ee.entropy, x1, k=k, base=NATS)),
                rtol=KNN_RTOL,
                k=k,
            )
        )
        out.append(
            case(
                f"ksgMutualInformation/1d/k={k}",
                "ksgMutualInformation",
                [x1, y1],
                np.array(seeded(ee.mi, x1, y1, k=k, base=NATS)),
                rtol=KNN_RTOL,
                k=k,
            )
        )
    out.append(
        case(
            "kozachenkoLeonenkoEntropy/2d/base=2",
            "kozachenkoLeonenkoEntropy",
            [x2],
            np.array(seeded(ee.entropy, x2, k=4, base=2)),
            rtol=KNN_RTOL,
            k=4,
            base=2,
        )
    )
    out.append(
        case(
            "ksgMutualInformation/2d",
            "ksgMutualInformation",
            [x2, y2],
            np.array(seeded(ee.mi, x2, y2, k=3, base=NATS)),
            rtol=KNN_RTOL,
            k=3,
        )
    )
    # sklearn's KSG (mutual_info_regression without its scaling and jitter) on tie-free data.
    out.append(
        case(
            "ksgMutualInformation/sklearn",
            "ksgMutualInformation",
            [x1[:, 0], y1[:, 0]],
            np.array(_compute_mi_cc(x1[:, 0], y1[:, 0], 3)),
            rtol=1e-10,
            k=3,
        )
    )
    out.append(
        case(
            "ksgConditionalMutualInformation",
            "ksgConditionalMutualInformation",
            [x1, y1, z1],
            np.array(seeded(ee.cmi, x1, y1, z1, k=3, base=NATS)),
            rtol=KNN_RTOL,
            k=3,
        )
    )
    q2 = g.normal(size=(250, 2)) * 1.5 + 0.3
    out.append(
        case(
            "knnKlDivergence",
            "knnKlDivergence",
            [x2, q2],
            np.array(ee.kldiv(x2, q2, k=3, base=NATS)),
            rtol=1e-10,
            k=3,
        )
    )
    labels = g.integers(0, 3, size=n).astype(np.float64)
    c = labels + g.normal(size=n)
    out.append(
        case(
            "mixedMutualInformation",
            "mixedMutualInformation",
            [c, labels],
            np.array(_compute_mi_cd(c, labels, 3)),
            rtol=1e-10,
            k=3,
        )
    )
    # Kernel estimators against scipy.stats.gaussian_kde.
    s1 = g.normal(size=(80, 1))
    s2 = np.c_[s1, s1 + 0.4 * g.normal(size=(80, 1))]
    out.append(case("kdeEntropy/1d", "kdeEntropy", [s1], np.array(kde_entropy(s1)), rtol=1e-10))
    out.append(case("kdeEntropy/2d", "kdeEntropy", [s2], np.array(kde_entropy(s2)), rtol=1e-10))
    kde_mi = kde_entropy(s2[:, :1]) + kde_entropy(s2[:, 1:]) - kde_entropy(s2)
    out.append(
        case("kdeMutualInformation", "kdeMutualInformation", [s2[:, :1], s2[:, 1:]], np.array(kde_mi), rtol=1e-9)
    )
    # Plug-in transfer entropy: I(X_{t+1}; Y_t | X_t^(k)) from pyitlib's conditional mutual information.
    src = g.integers(0, 2, size=200)
    noise = g.random(200) < 0.1
    tgt = np.r_[0, src[:-1]] ^ noise.astype(np.int64)
    te1 = drv.information_mutual_conditional(tgt[1:], src[:-1], tgt[:-1], base=NATS)
    out.append(case("transferEntropy/k=1", "transferEntropy", [src, tgt], np.array(te1)))
    hist2 = tgt[1:-1] * 2 + tgt[:-2]
    te2 = drv.information_mutual_conditional(tgt[2:], src[1:-1], hist2, base=2)
    out.append(case("transferEntropy/k=2", "transferEntropy", [src, tgt], np.array(te2), history=2, base=2))
    back = drv.information_mutual_conditional(src[1:], tgt[:-1], src[:-1], base=NATS)
    out.append(case("transferEntropy/reverse", "transferEntropy", [tgt, src], np.array(back)))
    # Variational bounds from a critic score matrix (positives on the diagonal).
    scores = g.normal(size=(6, 6)) + 2 * np.eye(6)
    off = ~np.eye(6, dtype=bool)
    nce = np.mean(np.diag(scores) - sp.logsumexp(scores, axis=1)) + np.log(6)
    nwj = np.mean(np.diag(scores)) - np.mean(np.exp(scores[off] - 1))
    dv = np.mean(np.diag(scores)) - (sp.logsumexp(scores[off]) - np.log(30))
    out.append(case("infoNce", "infoNce", [scores], np.array(nce)))
    out.append(case("nwjBound", "nwjBound", [scores], np.array(nwj)))
    out.append(case("mineBound", "mineBound", [scores], np.array(dv)))
    return out
