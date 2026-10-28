"""Naive Bayes against sklearn.naive_bayes: the Gaussian, multinomial, Bernoulli, categorical and complement families
with their smoothing, binarisation, normalisation, priors and sample weights, compared by class priors and joint
log-likelihood-derived posteriors."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def counts(g: np.random.Generator, sizes: list[int], d: int) -> tuple[FloatArray, FloatArray]:
    """Word-count style data: each class has its own Poisson rates."""
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for c, n in enumerate(sizes):
        rates = g.uniform(0.2, 3.0, size=d)
        xs.append(g.poisson(rates, size=(n, d)).astype(np.float64))
        ys.append(np.full(n, float(c)))
    return np.vstack(xs), np.concatenate(ys)


def conditional_mi(x: FloatArray, y: FloatArray, i: int, j: int) -> float:
    """I(X_i; X_j | C) in nats from maximum-likelihood probabilities."""
    n = len(y)
    total = 0.0
    for c in np.unique(y):
        xc = x[y == c]
        nc = len(xc)
        for a in np.unique(x[:, i]):
            for b in np.unique(x[:, j]):
                nab = float(np.sum((xc[:, i] == a) & (xc[:, j] == b)))
                if nab > 0:
                    na = float(np.sum(xc[:, i] == a))
                    nb = float(np.sum(xc[:, j] == b))
                    total += nab / n * np.log(nab * nc / (na * nb))
    return total


def tan_edges(x: FloatArray, y: FloatArray, root: int) -> tuple[FloatArray, FloatArray]:
    """Prim's tree over the conditional mutual information: the edges (parent, child) and their weights."""
    d = x.shape[1]
    w = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            w[i, j] = w[j, i] = conditional_mi(x, y, i, j)
    inside = [root]
    edges: list[tuple[float, float]] = []
    weights: list[float] = []
    while len(inside) < d:
        best = max(((w[u, v], -u, -v) for u in inside for v in range(d) if v not in inside))
        edges.append((float(-best[1]), float(-best[2])))
        weights.append(float(best[0]))
        inside.append(int(-best[2]))
    return np.array(edges), np.array(weights)


def tan_proba(x: FloatArray, y: FloatArray, q: FloatArray, edges: FloatArray, alpha: float) -> FloatArray:
    """Smoothed-count posteriors of the tree-augmented naive Bayes with the given parents."""
    classes = np.unique(y)
    d = x.shape[1]
    ncat = [int(max(x[:, j].max(), 0)) + 1 for j in range(d)]
    parent = {int(c): int(p) for p, c in edges}
    out = np.zeros((len(q), len(classes)))
    for ci, c in enumerate(classes):
        xc = x[y == c]
        prior = (len(xc) + alpha) / (len(y) + alpha * len(classes))
        for r, row in enumerate(q):
            score = np.log(prior)
            for j in range(d):
                if j in parent:
                    p = parent[j]
                    given = float(np.sum(xc[:, p] == row[p]))
                    joint = float(np.sum((xc[:, p] == row[p]) & (xc[:, j] == row[j])))
                else:
                    given = float(len(xc))
                    joint = float(np.sum(xc[:, j] == row[j]))
                score += np.log((joint + alpha) / (given + alpha * ncat[j]))
            out[r, ci] = score
    out = np.exp(out - out.max(axis=1, keepdims=True))
    return out / out.sum(axis=1, keepdims=True)


def aode_proba(x: FloatArray, y: FloatArray, q: FloatArray, alpha: float, min_count: int) -> FloatArray:
    """The AODE posterior: the average over qualifying parents of P(c, x_i) prod_j P(x_j | c, x_i)."""
    classes = np.unique(y)
    d = x.shape[1]
    k = len(classes)
    n = len(y)
    ncat = [int(x[:, j].max()) + 1 for j in range(d)]
    out = np.zeros((len(q), k))
    for r, row in enumerate(q):
        parents = [i for i in range(d) if np.sum(x[:, i] == row[i]) >= min_count]
        for ci, c in enumerate(classes):
            xc = x[y == c]
            terms = []
            for i in parents:
                nci = float(np.sum(xc[:, i] == row[i]))
                t = np.log((nci + alpha / (k * ncat[i])) / (n + alpha))
                for j in range(d):
                    if j != i:
                        ncij = float(np.sum((xc[:, i] == row[i]) & (xc[:, j] == row[j])))
                        t += np.log((ncij + alpha / ncat[j]) / (nci + alpha))
                terms.append(t)
            out[r, ci] = np.exp(terms).mean()
    return out / out.sum(axis=1, keepdims=True)


def cases() -> list[Case]:
    g = rng(1618)
    out: list[Case] = []
    x, y = counts(g, [22, 15, 9], 6)
    q = g.poisson(1.5, size=(10, 6)).astype(np.float64)
    xc = x + g.normal(size=x.shape) * 0.3
    qc = q + g.normal(size=q.shape) * 0.3
    w = g.uniform(0.5, 2.0, size=len(y))
    for name, smoothing, weights in [("plain", 1e-9, None), ("smoothed", 1e-2, None), ("weighted", 1e-9, w)]:
        nb = GaussianNB(var_smoothing=smoothing).fit(xc, y, sample_weight=weights)
        inputs = [xc, y, qc] if weights is None else [xc, y, qc, weights]
        out.append(
            case(
                f"gaussian/{name}",
                "nb",
                inputs,
                {"proba": nb.predict_proba(qc), "predict": nb.predict(qc), "logPrior": np.log(nb.class_prior_)},
                rtol=1e-9,
                atol=1e-10,
                family="gaussian",
                varSmoothing=smoothing,
            )
        )
    for alpha, fit_prior in [(1.0, True), (0.3, True), (2.0, False)]:
        nb = MultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(x, y)
        out.append(
            case(
                f"multinomial/{alpha}/{fit_prior}",
                "nb",
                [x, y, q],
                {"proba": nb.predict_proba(q), "predict": nb.predict(q), "logPrior": nb.class_log_prior_},
                rtol=1e-9,
                atol=1e-10,
                family="multinomial",
                alpha=alpha,
                fitPrior=fit_prior,
            )
        )
    nb = MultinomialNB(alpha=0.5).fit(x, y, sample_weight=w)
    out.append(
        case(
            "multinomial/weighted",
            "nb",
            [x, y, q, w],
            {"proba": nb.predict_proba(q), "predict": nb.predict(q), "logPrior": nb.class_log_prior_},
            rtol=1e-9,
            atol=1e-10,
            family="multinomial",
            alpha=0.5,
        )
    )
    for alpha, binarize in [(1.0, 0.0), (0.5, 1.5), (1.0, 3.0)]:
        nb = BernoulliNB(alpha=alpha, binarize=binarize).fit(x, y)
        out.append(
            case(
                f"bernoulli/{alpha}/{binarize}",
                "nb",
                [x, y, q],
                {"proba": nb.predict_proba(q), "predict": nb.predict(q), "logPrior": nb.class_log_prior_},
                rtol=1e-9,
                atol=1e-10,
                family="bernoulli",
                alpha=alpha,
                binarize=binarize,
            )
        )
    xi = np.minimum(x, 4.0)
    qi = np.minimum(q, 4.0)
    for alpha, min_cat in [(1.0, None), (0.5, 7)]:
        nb = CategoricalNB(alpha=alpha, min_categories=min_cat).fit(xi, y)
        params: dict[str, object] = {"family": "categorical", "alpha": alpha}
        if min_cat is not None:
            params["minCategories"] = min_cat
        out.append(
            case(
                f"categorical/{alpha}/{min_cat}",
                "nb",
                [xi, y, qi],
                {"proba": nb.predict_proba(qi), "predict": nb.predict(qi), "logPrior": nb.class_log_prior_},
                rtol=1e-9,
                atol=1e-10,
                **params,
            )
        )
    for alpha, norm in [(1.0, False), (0.5, True)]:
        nb = ComplementNB(alpha=alpha, norm=norm).fit(x, y)
        out.append(
            case(
                f"complement/{alpha}/{norm}",
                "nb",
                [x, y, q],
                {"proba": nb.predict_proba(q), "predict": nb.predict(q), "logPrior": nb.class_log_prior_},
                rtol=1e-9,
                atol=1e-10,
                family="complement",
                alpha=alpha,
                norm=norm,
            )
        )
    # Discrete features with planted dependences for the tree-augmented and averaged one-dependence classifiers.
    n = 80
    yb = (np.arange(n) % 2).astype(np.float64)
    f0 = np.where(g.uniform(size=n) < 0.8, yb, 1 - yb)
    f1 = np.where(g.uniform(size=n) < 0.85, f0, 1 - f0)
    f2 = g.integers(0, 3, size=n).astype(np.float64)
    f3 = np.where(g.uniform(size=n) < 0.7, f2 % 2, g.integers(0, 2, size=n)).astype(np.float64)
    xd = np.column_stack([f0, f1, f2, f3])
    qd = np.array([[a, b, c, e] for a in (0, 1) for b in (0, 1) for c in (0, 2) for e in (0, 1)], dtype=np.float64)
    edges, weights = tan_edges(xd, yb, 0)
    for alpha in [1.0, 0.25]:
        out.append(
            case(
                f"tan/{alpha}",
                "tan",
                [xd, yb, qd],
                {"edges": edges, "weights": weights, "proba": tan_proba(xd, yb, qd, edges, alpha)},
                rtol=1e-9,
                atol=1e-10,
                alpha=alpha,
                root=0,
            )
        )
        for min_count in [1, 30]:
            out.append(
                case(
                    f"aode/{alpha}/{min_count}",
                    "aode",
                    [xd, yb, qd],
                    {"proba": aode_proba(xd, yb, qd, alpha, min_count)},
                    rtol=1e-9,
                    atol=1e-10,
                    alpha=alpha,
                    minCount=min_count,
                )
            )
    return out
