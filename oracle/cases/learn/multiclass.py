"""Multiclass reductions against sklearn.multiclass over a logistic regression base: one-versus-rest (decision
function, probabilities, the binary special case), one-versus-one (votes plus scaled confidences), and error-correcting
output codes and nested dichotomies with given code books and hierarchies, formed from scikit-learn's own pieces (its
fitted binary models, its distance decoding and the product of branch probabilities)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def base(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, tol=1e-12, max_iter=20000)


def blobs(g: np.random.Generator, sizes: list[int], d: int) -> tuple[FloatArray, FloatArray]:
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for c, n in enumerate(sizes):
        xs.append(g.normal(size=(n, d)) + 2.2 * g.normal(size=d))
        ys.append(np.full(n, float(c)))
    return np.vstack(xs), np.concatenate(ys)


def cases() -> list[Case]:
    g = rng(8675309)
    out: list[Case] = []
    for name, sizes in [("binary", [20, 15]), ("three", [18, 14, 12]), ("four", [14, 12, 10, 9])]:
        x, y = blobs(g, sizes, 3)
        q = np.vstack([g.normal(size=(8, 3)) * 2, x[:2]])
        ovr = OneVsRestClassifier(base(1.0)).fit(x, y)
        out.append(
            case(
                f"ovr/{name}",
                "ovr",
                [x, y, q],
                {"predict": ovr.predict(q), "decision": ovr.decision_function(q), "proba": ovr.predict_proba(q)},
                rtol=1e-6,
                atol=1e-7,
                C=1.0,
            )
        )
        ovo = OneVsOneClassifier(base(0.5)).fit(x, y)
        out.append(
            case(
                f"ovo/{name}",
                "ovo",
                [x, y, q],
                {"predict": ovo.predict(q), "decision": ovo.decision_function(q)},
                rtol=1e-7,
                atol=1e-8,
                C=0.5,
            )
        )
        if name == "binary":
            continue
        k = len(sizes)
        book = (g.uniform(size=(k, 2 * k)) > 0.5).astype(np.float64)
        book[0, book.sum(axis=0) == 0] = 1.0
        book[0, book.sum(axis=0) == k] = 0.0
        codes = book[y.astype(int)]
        models = [base(1.0).fit(x, codes[:, j]) for j in range(book.shape[1])]
        scores = np.column_stack([m.decision_function(q) for m in models])
        signed = 2 * book - 1
        dist = np.sqrt(((scores[:, None, :] - signed[None, :, :]) ** 2).sum(axis=2))
        out.append(
            case(
                f"ecoc/{name}",
                "ecoc",
                [x, y, q, book],
                {"predict": np.argmin(dist, axis=1).astype(np.float64), "decision": -dist},
                rtol=1e-7,
                atol=1e-8,
                C=1.0,
            )
        )
        # A nested dichotomy: a hierarchy over the classes, the probability of a class the product on its path.
        classes = list(range(k))
        nodes: list[tuple[list[int], list[int]]] = []

        def grow(s: list[int], nodes: list[tuple[list[int], list[int]]] = nodes) -> None:
            if len(s) < 2:
                return
            cut = max(1, len(s) // 2)
            nodes.append((s[:cut], s[cut:]))
            grow(s[:cut], nodes)
            grow(s[cut:], nodes)

        grow(classes)
        proba = np.ones((len(q), k))
        for left, right in nodes:
            rows = np.isin(y, left + right)
            fit = base(1.0).fit(x[rows], np.isin(y[rows], right).astype(np.float64))
            pr = fit.predict_proba(q)[:, 1]
            for c in right:
                proba[:, c] *= pr
            for c in left:
                proba[:, c] *= 1 - pr
        width = max(len(n[0]) for n in nodes) + max(len(n[1]) for n in nodes)
        hierarchy = np.array(
            [(left + [-1] * (width - len(left)), right + [-1] * (width - len(right))) for left, right in nodes],
            dtype=np.float64,
        )
        out.append(
            case(
                f"nested/{name}",
                "nested",
                [x, y, q, hierarchy],
                {"proba": proba, "predict": np.argmax(proba, axis=1).astype(np.float64)},
                rtol=1e-7,
                atol=1e-8,
                C=1.0,
            )
        )
    return out
