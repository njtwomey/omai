"""Plug-in, kernel and transfer-entropy estimators and variational bounds against scipy, sklearn, pyitlib and NPEET."""

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
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from pyitlib import discrete_random_variable as drv  # pyright: ignore[reportMissingTypeStubs]

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
    # Continuous transfer entropy: NPEET's conditional mutual information on the lagged series.
    ys = g.normal(size=150)
    xs = np.r_[0.0, 0.8 * ys[:-1]] + 0.5 * g.normal(size=150)
    te_ksg = seeded(ee.cmi, xs[1:, None], ys[:-1, None], xs[:-1, None], k=3, base=NATS)
    out.append(
        case(
            "transferEntropy/ksg",
            "transferEntropy",
            [ys, xs],
            np.array(te_ksg),
            rtol=1e-7,
            estimator="ksg",
            neighbours=3,
        )
    )
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
