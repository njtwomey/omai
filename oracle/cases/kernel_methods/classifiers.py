"""Kernel classifiers against scikit-learn: kernel logistic regression through the identity with L2-penalised logistic
regression in a feature space (the linear kernel directly, an RBF kernel through Nystroem with every row a landmark,
where `|w|^2 = alpha' K alpha`), the least-squares SVM through numpy's solve of its dual linear system, and kernel ridge
classification through RidgeClassifier (linear kernel, no intercept) and KernelRidge on one-versus-rest plus and minus
one targets."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics.pairwise import rbf_kernel

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def blobs(g: np.random.Generator, n: int, k: int) -> tuple[FloatArray, FloatArray]:
    centres = np.array([[0.0, 0.0], [2.5, 0.0], [0.0, 2.5]])[:k]
    labels = np.repeat(np.arange(k), n)
    x = centres[labels] + g.normal(size=(n * k, 2))
    return x, labels.astype(np.float64)


def cases() -> list[Case]:
    g = rng(909)
    x, y = blobs(g, 20, 2)
    xt, _ = blobs(g, 8, 2)
    x3, y3 = blobs(g, 15, 3)
    xt3, _ = blobs(g, 6, 3)
    out: list[Case] = []
    lam = 0.5
    lin = LogisticRegression(C=1.0 / lam, tol=1e-12, max_iter=10000).fit(x, y)
    out.append(
        case(
            "klr/linear",
            "klr",
            [x, y, xt],
            {"decision": lin.decision_function(xt)},
            rtol=1e-5,
            atol=1e-6,
            kernel="linear",
            lam=lam,
            gamma=1.0,
        )
    )
    gamma = 0.3
    ny = Nystroem(kernel="rbf", gamma=gamma, n_components=len(x), random_state=0).fit(x)
    rbf = LogisticRegression(C=1.0 / lam, tol=1e-12, max_iter=10000).fit(ny.transform(x), y)
    out.append(
        case(
            "klr/rbf",
            "klr",
            [x, y, xt],
            {"decision": rbf.decision_function(ny.transform(xt))},
            rtol=1e-4,
            atol=1e-5,
            kernel="rbf",
            lam=lam,
            gamma=gamma,
        )
    )
    c = 4.0
    k = rbf_kernel(x, x, gamma=gamma)
    n = len(x)
    system = np.zeros((n + 1, n + 1))
    system[0, 1:] = 1.0
    system[1:, 0] = 1.0
    system[1:, 1:] = k + np.eye(n) / c
    rhs = np.concatenate([[0.0], np.where(y == 1, 1.0, -1.0)])
    sol = np.linalg.solve(system, rhs)
    out.append(
        case(
            "lssvm/rbf",
            "lssvm",
            [x, y, xt],
            {"dual": sol[1:], "bias": sol[:1], "decision": rbf_kernel(xt, x, gamma=gamma) @ sol[1:] + sol[0]},
            rtol=1e-9,
            atol=1e-10,
            gamma=gamma,
            c=c,
        )
    )
    alpha = 0.8
    ridge = RidgeClassifier(alpha=alpha, fit_intercept=False).fit(x3, y3)
    out.append(
        case(
            "ridge-classifier/linear",
            "ridgeClassifier",
            [x3, y3, xt3],
            {"decision": ridge.decision_function(xt3)},
            rtol=1e-8,
            atol=1e-9,
            kernel="linear",
            alpha=alpha,
            gamma=1.0,
        )
    )
    targets = -np.ones((len(x3), 3))
    targets[np.arange(len(x3)), y3.astype(int)] = 1.0
    krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma).fit(x3, targets)  # pyright: ignore[reportArgumentType]
    out.append(
        case(
            "ridge-classifier/rbf",
            "ridgeClassifier",
            [x3, y3, xt3],
            {"decision": krr.predict(xt3)},
            rtol=1e-8,
            atol=1e-9,
            kernel="rbf",
            alpha=alpha,
            gamma=gamma,
        )
    )
    return out
