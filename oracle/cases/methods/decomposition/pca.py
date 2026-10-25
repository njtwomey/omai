"""PCA and incremental PCA against scikit-learn: axes, explained variance, ratios, noise variance and scores."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.decomposition import PCA, IncrementalPCA

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-10}


def data() -> FloatArray:
    g = rng(21)
    mix = np.array([[2.0, 0.5, 0.1, 0.0], [0.3, 1.0, 0.2, 0.1], [0.0, 0.2, 0.5, 0.3], [0.1, 0.0, 0.1, 0.2]])
    return g.standard_normal((40, 4)) @ mix + np.array([1.0, -2.0, 0.5, 3.0])


def cases() -> list[Case]:
    out: list[Case] = []
    x = data()
    for k in [2, 4]:
        p = PCA(n_components=k, svd_solver="full").fit(x)
        out.append(case(f"pca{k}/components", "components", [x], np.asarray(p.components_), **TOL, k=k))
        out.append(case(f"pca{k}/variance", "variance", [x], np.asarray(p.explained_variance_), **TOL, k=k))
        out.append(case(f"pca{k}/ratio", "ratio", [x], np.asarray(p.explained_variance_ratio_), **TOL, k=k))
        out.append(case(f"pca{k}/noise", "noise", [x], np.float64(p.noise_variance_), **TOL, k=k))
        out.append(case(f"pca{k}/scores", "scores", [x], np.asarray(p.transform(x)), **TOL, k=k))
        out.append(case(f"pca{k}/score", "score", [x], np.float64(p.score(x)), **TOL, k=k))
    w = PCA(n_components=2, whiten=True).fit(x)
    out.append(case("pca2/whitened", "whitened", [x], np.asarray(w.transform(x)), **TOL, k=2))
    ip = IncrementalPCA(n_components=2, batch_size=12).fit(x)
    out.append(case("ipca/components", "ipcaComponents", [x], np.asarray(ip.components_), **TOL, k=2, batch=12))
    out.append(case("ipca/variance", "ipcaVariance", [x], np.asarray(ip.explained_variance_), **TOL, k=2, batch=12))
    out.append(case("ipca/ratio", "ipcaRatio", [x], np.asarray(ip.explained_variance_ratio_), **TOL, k=2, batch=12))
    out.append(case("ipca/noise", "ipcaNoise", [x], np.float64(ip.noise_variance_), **TOL, k=2, batch=12))
    return out
