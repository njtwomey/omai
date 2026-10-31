"""Discriminant analysis against sklearn.discriminant_analysis: linear discriminants by the SVD, least-squares and
eigen solvers (with fixed and Ledoit-Wolf shrinkage, priors, the discriminant axes and their explained variance) and
quadratic discriminants with the eigenvalue regularisation."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def blobs(g: np.random.Generator, sizes: list[int], d: int) -> tuple[FloatArray, FloatArray]:
    """Gaussian classes with different means and a shared correlated covariance."""
    mix = g.normal(size=(d, d)) * 0.5 + np.eye(d)
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for c, n in enumerate(sizes):
        mean = g.normal(size=d) * 1.6
        xs.append(g.normal(size=(n, d)) @ mix + mean)
        ys.append(np.full(n, float(c)))
    return np.vstack(xs), np.concatenate(ys)


def axes(scalings: FloatArray, q: int) -> FloatArray:
    """The first q discriminant axes with the largest-magnitude entry of each made positive."""
    out = np.array(scalings[:, :q], dtype=np.float64)
    for j in range(out.shape[1]):
        if out[np.argmax(np.abs(out[:, j])), j] < 0:
            out[:, j] *= -1
    return out


def cases() -> list[Case]:
    g = rng(31415)
    out: list[Case] = []
    x, y = blobs(g, [25, 18, 12], 4)
    q = np.vstack([g.normal(size=(9, 4)) * 1.5 + 1.0, x[:3] + 0.05])
    for name, solver, shrinkage in [
        ("svd", "svd", None),
        ("lsqr", "lsqr", None),
        ("lsqr/shrink0.3", "lsqr", 0.3),
        ("lsqr/auto", "lsqr", "auto"),
        ("eigen", "eigen", None),
        ("eigen/shrink0.1", "eigen", 0.1),
        ("eigen/auto", "eigen", "auto"),
    ]:
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage).fit(x, y)  # pyright: ignore[reportArgumentType]
        dec = lda.decision_function(q)
        output: dict[str, FloatArray] = {
            "predict": np.asarray(lda.predict(q), dtype=np.float64),
            "proba": lda.predict_proba(q),
            "decisionDiff": dec[:, 1:] - dec[:, :1],
            "means": lda.means_,
            "priors": lda.priors_,
        }
        # With shrinkage the eigen solver forms Sb as St - Sw of two shrunk covariances; the axes are not compared.
        if solver != "lsqr" and shrinkage is None:
            output["explained"] = lda.explained_variance_ratio_
            output["scalings"] = axes(lda.scalings_, 2)
        params: dict[str, object] = {}
        if shrinkage is not None:
            params["shrinkage"] = shrinkage
        out.append(case(f"lda/{name}", "lda", [x, y, q], output, rtol=1e-7, atol=1e-8, **params))
    # Binary: the decision function itself.
    xb, yb = blobs(g, [30, 20], 3)
    qb = g.normal(size=(8, 3)) * 1.5 + 1.0
    for solver in ["svd", "lsqr"]:
        lda = LinearDiscriminantAnalysis(solver=solver).fit(xb, yb)  # pyright: ignore[reportArgumentType]
        binary: dict[str, FloatArray] = {
            "predict": np.asarray(lda.predict(qb), dtype=np.float64),
            "proba": lda.predict_proba(qb),
            "decisionDiff": lda.decision_function(qb)[:, None],
        }
        if solver == "svd":
            binary["scalings"] = axes(lda.scalings_, 1)
        out.append(case(f"lda/binary/{solver}", "lda", [xb, yb, qb], binary, rtol=1e-7, atol=1e-8))
    priors = np.array([0.2, 0.5, 0.3])
    lda = LinearDiscriminantAnalysis(solver="svd", priors=priors).fit(x, y)
    dec = lda.decision_function(q)
    out.append(
        case(
            "lda/priors",
            "lda",
            [x, y, q],
            {
                "predict": np.asarray(lda.predict(q), dtype=np.float64),
                "proba": lda.predict_proba(q),
                "decisionDiff": dec[:, 1:] - dec[:, :1],
            },
            rtol=1e-7,
            atol=1e-8,
            priors=priors,
        )
    )
    for reg in [0.0, 0.2, 0.7]:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg).fit(x, y)
        out.append(
            case(
                f"qda/reg{reg}",
                "qda",
                [x, y, q],
                {"predict": qda.predict(q), "proba": qda.predict_proba(q), "means": qda.means_, "priors": qda.priors_},
                rtol=1e-7,
                atol=1e-8,
                regParam=reg,
            )
        )
    for shrinkage in [0.2, "auto"]:
        qda = QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"qda/eigen/{shrinkage}",
                "qda",
                [x, y, q],
                {"predict": qda.predict(q), "proba": qda.predict_proba(q), "means": qda.means_},
                rtol=1e-7,
                atol=1e-8,
                shrinkage=shrinkage,
            )
        )
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1, priors=priors).fit(x, y)
    out.append(
        case(
            "qda/priors",
            "qda",
            [x, y, q],
            {"predict": qda.predict(q), "proba": qda.predict_proba(q)},
            rtol=1e-7,
            atol=1e-8,
            regParam=0.1,
            priors=priors,
        )
    )
    return out
