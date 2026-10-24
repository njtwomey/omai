"""Discriminant analysis against scikit-learn: LDA with the svd and lsqr solvers (with and without shrinkage), its
explained variance ratios, and QDA with and without regularisation."""

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(81)
    y = (np.arange(90) % 3).astype(np.float64)
    a = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]])
    x = g.normal(size=(90, 3)) @ a + np.array([[0, 0, 0], [1.5, 0.5, 0], [0, 1.2, 0.8]])[y.astype(int)]
    q = x[:7]
    out: list[Case] = []
    svd = LinearDiscriminantAnalysis().fit(x, y)
    out.append(
        case(
            "lda/svd",
            "lda",
            [x, y],
            {
                "coef": svd.coef_,
                "intercept": svd.intercept_,
                "proba": svd.predict_proba(q),
                "ratio": svd.explained_variance_ratio_,
            },
            rtol=1e-9,
            atol=1e-10,
            solver="svd",
        )
    )
    for shrinkage in [None, 0.3, "auto"]:
        lsqr = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(x, y)
        out.append(
            case(
                f"lda/lsqr/{shrinkage}",
                "lda",
                [x, y],
                {"coef": lsqr.coef_, "intercept": lsqr.intercept_, "proba": lsqr.predict_proba(q)},
                rtol=1e-9,
                atol=1e-10,
                solver="lsqr",
                shrinkage="none" if shrinkage is None else shrinkage,
            )
        )
    for reg in [0.0, 0.2]:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg).fit(x, y)
        out.append(case(f"qda/{reg}", "qda", [x, y], {"proba": qda.predict_proba(q)}, rtol=1e-9, atol=1e-10, reg=reg))
    return out
