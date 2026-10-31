"""Dual coordinate descent for the linear SVM against liblinear through scikit-learn's LinearSVC (dual form)."""

import warnings
from typing import cast

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.svm import LinearSVC

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    """Weights and intercept of the L1-loss (hinge) and L2-loss (squared hinge) problems on overlapping blobs.

    liblinear regularises the intercept with the weights (intercept_scaling = 1), which the TypeScript solver does
    too, so the primal solution is unique and both solvers must reach it. The oracle refuses a case where liblinear
    hits its iteration cap (it does at large C with a tight tolerance, and then its answer is not the optimum).
    """
    g = rng(31)
    out: list[Case] = []
    x = np.vstack([g.normal(-0.6, 1.0, size=(30, 3)), g.normal(0.6, 1.0, size=(30, 3))])
    y = np.where(np.arange(60) < 30, -1.0, 1.0)
    for loss, sk_loss in [("l1", "hinge"), ("l2", "squared_hinge")]:
        for c in [0.05, 0.3, 1.0, 3.0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LinearSVC(
                    C=c,
                    loss=sk_loss,
                    dual=cast("str", True),  # the stub types dual as str; True selects the dual solver
                    tol=1e-9,
                    max_iter=1_000_000,
                    random_state=0,
                ).fit(x, y)
            if int(np.max(clf.n_iter_)) >= 1_000_000:
                raise RuntimeError(f"liblinear did not converge for {loss} at C={c}")
            output = {"coef": clf.coef_.ravel(), "intercept": np.asarray(clf.intercept_).ravel()[0]}
            out.append(case(f"{loss}/C{c}", "linearSvmDualCd", [x, y], output, rtol=1e-5, atol=1e-5, C=c, loss=loss))
    return out
