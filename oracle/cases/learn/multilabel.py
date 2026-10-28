"""Multilabel wrappers against scikit-learn: binary relevance as one-versus-rest on a label matrix, classifier chains
with given orders and chain methods, and the label powerset formed from a multiclass Gaussian naive Bayes on the
distinct label combinations."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def base() -> LogisticRegression:
    return LogisticRegression(C=1.0, tol=1e-12, max_iter=20000)


def cases() -> list[Case]:
    g = rng(77)
    out: list[Case] = []
    n = 70
    x = g.normal(size=(n, 3))
    z = x @ g.normal(size=(3, 3)) + 0.4 * g.normal(size=(n, 3))
    y = (z > 0).astype(np.float64)
    y[:, 2] = np.where(g.uniform(size=n) < 0.8, y[:, 0], 1 - y[:, 0])  # a label correlated with the first
    q = g.normal(size=(9, 3)) * 1.5
    ovr = OneVsRestClassifier(base()).fit(x, y)
    out.append(
        case(
            "relevance",
            "relevance",
            [x, y, q],
            {"predict": ovr.predict(q), "proba": ovr.predict_proba(q), "decision": ovr.decision_function(q)},
            rtol=1e-6,
            atol=1e-7,
        )
    )
    for order in ([2, 0, 1], [0, 1, 2]):
        for method in ["predict", "predict_proba", "decision_function"]:
            chain = ClassifierChain(base(), order=order, chain_method=method).fit(x, y)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"chain/{order}/{method}",
                    "chain",
                    [x, y, q],
                    {"predict": chain.predict(q), "proba": chain.predict_proba(q)},
                    rtol=1e-6,
                    atol=1e-7,
                    order=np.array(order, dtype=np.float64),
                    method=method,
                )
            )
    xp = g.normal(size=(n, 2)) * 2
    yp = np.column_stack([xp[:, 0] > 0, xp[:, 1] > 0, (xp[:, 0] + xp[:, 1]) > 1]).astype(np.float64)
    qp = g.normal(size=(8, 2)) * 2
    combos = np.unique(yp, axis=0)
    index = np.array([int(np.where((combos == row).all(axis=1))[0][0]) for row in yp], dtype=np.float64)
    nb = GaussianNB().fit(xp, index)
    pc = np.zeros((len(qp), len(combos)))
    pc[:, nb.classes_.astype(int)] = nb.predict_proba(qp)
    marginal = pc @ combos
    predicted: FloatArray = combos[nb.predict(qp).astype(int)]
    out.append(
        case(
            "powerset",
            "powerset",
            [xp, yp, qp],
            {"predict": predicted, "proba": marginal, "combinations": pc},
            rtol=1e-9,
            atol=1e-10,
        )
    )
    return out
