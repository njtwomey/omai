"""Principal-component regression against scikit-learn (a pipeline of StandardScaler/PCA and LinearRegression, the
coefficients mapped back to the features) and reduced-rank regression against a numpy implementation from its
definition (least squares, then the SVD of the fitted values; the canonical weighting through the output covariance)."""

import numpy as np
import numpy.typing as npt
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

type Mat = npt.NDArray[np.float64]


def rrr(x: Mat, y: Mat, rank: int, alpha: float, weighting: str) -> tuple[Mat, Mat]:
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    full = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    fitted = xc @ full
    if weighting == "covariance":
        w, v = np.linalg.eigh(yc.T @ yc / len(y))
        half = v @ np.diag(w**0.5) @ v.T
        half_inv = v @ np.diag(w**-0.5) @ v.T
        vt = np.linalg.svd(fitted @ half_inv, full_matrices=False)[2]
        vr = vt[:rank].T
        coef = full @ half_inv @ vr @ vr.T @ half
    else:
        vt = np.linalg.svd(fitted, full_matrices=False)[2]
        vr = vt[:rank].T
        coef = full @ vr @ vr.T
    return coef, ym - xm @ coef


def cases() -> list[Case]:
    g = rng(1951)
    out: list[Case] = []
    n, d = 90, 5
    latent = g.normal(size=(n, 2))
    x = latent @ g.normal(size=(2, d)) + 0.3 * g.normal(size=(n, d)) * np.linspace(1.0, 4.0, d)
    y1 = x @ np.array([1.0, -0.5, 0.0, 2.0, 0.3]) + 0.4 + 0.3 * g.normal(size=n)
    for k in (1, 2, 5):
        for scale in (False, True):
            if scale:
                scaler = StandardScaler().fit(x)
                z = scaler.transform(x)
                std = np.asarray(scaler.scale_)
            else:
                z = x
                std = np.ones(d)
            pca = PCA(n_components=k, svd_solver="full").fit(z)
            lr = LinearRegression().fit(pca.transform(z), y1)
            coef = np.asarray(pca.components_).T @ np.asarray(lr.coef_) / std
            intercept = float(lr.intercept_) - float(np.mean(x, axis=0) @ coef)
            out.append(
                case(
                    f"pcr/k{k}/scale{int(scale)}",
                    "pcr",
                    [x, y1],
                    {
                        "coef": coef,
                        "intercept": np.array(intercept),
                        "explainedVariance": pca.explained_variance_,
                    },
                    rtol=1e-8,
                    atol=1e-9,
                    nComponents=k,
                    scale=1.0 if scale else 0.0,
                )
            )
    # Reduced-rank regression.
    n, d, q = 100, 4, 4
    x = g.normal(size=(n, d))
    b = np.outer(g.normal(size=d), g.normal(size=q)) + np.outer(g.normal(size=d), g.normal(size=q)) * 0.4
    y = x @ b + np.array([0.5, -1.0, 0.0, 2.0]) + g.normal(size=(n, q)) * np.array([0.3, 1.0, 0.5, 2.0])
    for rank in (1, 2, 4):
        for weighting in ("identity", "covariance"):
            for alpha in (0.0, 3.0):
                coef, intercept = rrr(x, y, rank, alpha, weighting)
                out.append(
                    case(
                        f"rrr/rank{rank}/{weighting}/alpha{alpha}",
                        "reducedRank",
                        [x, y],
                        {"coef": coef, "intercept": intercept},
                        rtol=1e-8,
                        atol=1e-9,
                        rank=rank,
                        alpha=alpha,
                        weighting=weighting,
                    )
                )
    return out
