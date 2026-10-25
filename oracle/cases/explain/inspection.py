"""Model inspection against sklearn.inspection (permutation importance, partial dependence and ICE, for a polynomial
regression pipeline and naive Bayes classifiers) and numpy references of accumulated local effects (the ALEPlot
recipe), Friedman's H-statistic and the second-order functional ANOVA, which sklearn does not provide."""

from typing import Any

import numpy as np
import sklearn
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def effective_permutations(seed: int, n: int, repeats: int) -> FloatArray:
    """The row permutation each repeat applies to the original column, as sklearn's cumulative shuffles do."""
    random_seed = np.random.RandomState(seed).randint(np.iinfo(np.int32).max + 1)
    rs = np.random.RandomState(random_seed)
    idx = np.arange(n)
    eff = np.arange(n)
    out = []
    for _ in range(repeats):
        rs.shuffle(idx)
        eff = eff[idx]
        out.append(eff.copy())
    return np.array(out, dtype=np.float64)


def ale(predict: Any, x: FloatArray, j: int, bins: int) -> tuple[FloatArray, FloatArray]:  # noqa: ANN401
    """First-order ALE with quantile bins, centred as R's ALEPlot."""
    edges = np.unique(np.quantile(x[:, j], np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, x[:, j], side="left") - 1, 0, len(edges) - 2)
    lo, hi = x.copy(), x.copy()
    lo[:, j] = edges[b]
    hi[:, j] = edges[b + 1]
    diff = predict(hi) - predict(lo)
    counts = np.bincount(b, minlength=len(edges) - 1)
    delta = np.bincount(b, weights=diff, minlength=len(edges) - 1) / counts
    acc = np.concatenate([[0.0], np.cumsum(delta)])
    centre = np.sum((acc[:-1] + acc[1:]) / 2 * counts) / len(x)
    return edges, acc - centre


def pd_at_rows(predict: Any, x: FloatArray, cols: list[int]) -> FloatArray:  # noqa: ANN401
    """Centred partial dependence on `cols` evaluated at every row."""
    out = np.empty(len(x))
    for i in range(len(x)):
        z = x.copy()
        z[:, cols] = x[i, cols]
        out[i] = predict(z).mean()
    return out - out.mean()


def cases() -> list[Case]:
    g = rng(4343)
    n = 40
    x = g.uniform(-2, 2, size=(n, 3))
    y = x[:, 0] ** 2 + x[:, 0] * x[:, 1] - x[:, 2] + 0.3 * g.normal(size=n)
    model = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())]).fit(x, y)
    out: list[Case] = []
    r: Any = permutation_importance(model, x, y, n_repeats=3, random_state=0, scoring="r2")
    out.append(
        case(
            "permutation",
            "permutation",
            [x, y, effective_permutations(0, n, 3)],
            {"importances": r.importances, "mean": r.importances_mean, "std": r.importances_std},
            rtol=1e-9,
            atol=1e-12,
        )
    )
    for features, kind in [([0], "both"), ([1], "average"), ([0, 1], "average")]:
        pd: Any = partial_dependence(model, x, features, grid_resolution=8, kind=kind)
        outputs: dict[str, object] = {
            "average": pd["average"].reshape(1, -1),
            "grid": np.concatenate(pd["grid_values"]),
        }
        if kind == "both":
            outputs["individual"] = pd["individual"][0]
        out.append(
            case(
                f"pd/regression/{features}",
                "pdRegression",
                [x, y],
                outputs,
                features=features,
                resolution=8,
                rtol=1e-10,
                atol=1e-10,
            )
        )
    yb = (x[:, 0] + x[:, 1] > 0).astype(np.float64)
    nb = GaussianNB().fit(x, yb)
    pd = partial_dependence(nb, x, [0], grid_resolution=6, kind="average")
    out.append(
        case(
            "pd/binary",
            "pdClassifier",
            [x, yb],
            {"average": pd["average"].reshape(1, -1)},
            features=[0],
            resolution=6,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    y3 = np.digitize(x[:, 0] - x[:, 2], [-1.0, 1.0]).astype(np.float64)
    nb3 = GaussianNB().fit(x, y3)
    pd = partial_dependence(nb3, x, [2], grid_resolution=5, kind="average")
    out.append(
        case(
            "pd/multiclass",
            "pdClassifier",
            [x, y3],
            {"average": pd["average"]},
            features=[2],
            resolution=5,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    xc = x.copy()
    xc[:, 1] = np.round(xc[:, 1])
    model_c = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())]).fit(xc, y)
    pd = partial_dependence(model_c, xc, [1], grid_resolution=20, kind="average")
    out.append(
        case(
            "pd/few-values",
            "pdRegression",
            [xc, y],
            {"average": pd["average"], "grid": pd["grid_values"][0]},
            features=[1],
            resolution=20,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    edges, effects = ale(model.predict, x, 0, 6)
    out.append(
        case("ale", "ale", [x, y], {"edges": edges, "effects": effects}, feature=0, bins=6, rtol=1e-10, atol=1e-10)
    )
    xs = x[:15]
    pj, pk, pjk = (
        pd_at_rows(model.predict, xs, [0]),
        pd_at_rows(model.predict, xs, [1]),
        pd_at_rows(model.predict, xs, [0, 1]),
    )
    h01 = np.sum((pjk - pj - pk) ** 2) / np.sum(pjk**2)
    f = model.predict(xs) - model.predict(xs).mean()
    rest = pd_at_rows(model.predict, xs, [1, 2])
    h0 = np.sum((f - pj - rest) ** 2) / np.sum(f**2)
    out.append(
        case(
            "hStatistic",
            "hStatistic",
            [x, y, xs],
            {"pair": np.array(h01), "total": np.array(h0)},
            rtol=1e-10,
            atol=1e-12,
        )
    )
    total = f.var()
    pz = pd_at_rows(model.predict, xs, [2])
    shares = [
        pj.var() / total,
        pk.var() / total,
        pz.var() / total,
        (pjk - pj - pk).var() / total,
        (pd_at_rows(model.predict, xs, [0, 2]) - pj - pz).var() / total,
        (pd_at_rows(model.predict, xs, [1, 2]) - pk - pz).var() / total,
    ]
    out.append(case("anova", "anova", [x, y, xs], {"shares": np.array(shares)}, rtol=1e-9, atol=1e-12))
    return out
