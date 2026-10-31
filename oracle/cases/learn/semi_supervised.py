"""Label propagation, label spreading and self-training against sklearn.semi_supervised: the RBF and neighbour graphs,
the clamped and soft-clamped diffusions with their label distributions and out-of-sample probabilities, and the
threshold and k-best selection of self-training (naive Bayes, logistic regression), with each row's round."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def blobs(g: np.random.Generator, sizes: list[int], spread: float) -> tuple[FloatArray, FloatArray]:
    """Gaussian clusters around well-separated centres, with their class labels."""
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for c, n in enumerate(sizes):
        centre = 3.0 * np.array([np.cos(2 * np.pi * c / len(sizes)), np.sin(2 * np.pi * c / len(sizes))])
        xs.append(centre + spread * g.normal(size=(n, 2)))
        ys.append(np.full(n, float(c)))
    return np.vstack(xs), np.concatenate(ys)


def hide(g: np.random.Generator, y: FloatArray, per_class: int) -> FloatArray:
    """The targets with all but `per_class` rows of each class replaced by -1."""
    out = np.full_like(y, -1.0)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        out[g.choice(rows, size=per_class, replace=False)] = c
    return out


def cases() -> list[Case]:
    g = rng(20261031)
    out: list[Case] = []
    for name, sizes, spread in [("two", [14, 12], 0.8), ("three", [10, 9, 11], 0.7)]:
        x, y = blobs(g, sizes, spread)
        yh = hide(g, y, 2)
        q = np.vstack([g.normal(size=(6, 2)) * 2.5, x[:2]])
        for kernel, params in [("rbf", {"gamma": 0.7}), ("knn", {"n_neighbors": 5})]:
            lp = LabelPropagation(kernel=kernel, max_iter=1000, tol=1e-6, **params).fit(x, yh)  # type: ignore[arg-type]
            out.append(
                case(
                    f"propagation/{kernel}/{name}",
                    "propagation",
                    [x, yh, q],
                    {
                        "distributions": lp.label_distributions_,
                        "transduction": lp.transduction_,
                        "proba": lp.predict_proba(q),
                        "predict": lp.predict(q),
                    },
                    rtol=1e-8,
                    atol=1e-9,
                    kernel=kernel,
                    maxIter=1000,
                    tol=1e-6,
                    **params,
                )
            )
            ls = LabelSpreading(kernel=kernel, alpha=0.4, max_iter=200, tol=1e-8, **params).fit(x, yh)  # type: ignore[arg-type]
            out.append(
                case(
                    f"spreading/{kernel}/{name}",
                    "spreading",
                    [x, yh, q],
                    {
                        "distributions": ls.label_distributions_,
                        "transduction": ls.transduction_,
                        "proba": ls.predict_proba(q),
                        "predict": ls.predict(q),
                    },
                    rtol=1e-8,
                    atol=1e-9,
                    kernel=kernel,
                    alpha=0.4,
                    maxIter=200,
                    tol=1e-8,
                    **params,
                )
            )
        # The k-best rule breaks ties between equal top probabilities arbitrarily (numpy's argpartition), and a naive
        # Bayes saturates at probability one on far rows, so that rule runs over a regularised logistic regression
        # (two classes only).
        for crit, params2 in [("threshold", {"threshold": 0.9}), ("k_best", {"k_best": 4})]:
            if crit == "k_best" and len(sizes) != 2:
                continue
            xs, ys = blobs(g, sizes, 1.6 if crit == "k_best" else 1.1)
            ysh = hide(g, ys, 3)
            base = LogisticRegression(C=1.0, tol=1e-12, max_iter=5000) if crit == "k_best" else GaussianNB()
            st = SelfTrainingClassifier(base, criterion=crit, max_iter=6, **params2).fit(xs, ysh)  # type: ignore[arg-type]
            out.append(
                case(
                    f"selftraining/{crit}/{name}",
                    "selftraining",
                    [xs, ysh, q],
                    {
                        "transduction": st.transduction_,
                        "iter": st.labeled_iter_,
                        "proba": st.predict_proba(q),
                        "predict": st.predict(q),
                    },
                    rtol=1e-6 if crit == "k_best" else 1e-7,
                    atol=1e-7 if crit == "k_best" else 1e-8,
                    base="logistic" if crit == "k_best" else "naive-bayes",
                    criterion=crit,
                    maxIter=6,
                    **{("kBest" if k == "k_best" else k): v for k, v in params2.items()},
                )
            )
    return out
