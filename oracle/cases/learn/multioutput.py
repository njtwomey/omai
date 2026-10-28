"""Multi-output wrappers against sklearn.multioutput: independent regressors, regressor chains with a given order, and
independent multiclass outputs (Gaussian naive Bayes per output)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, RegressorChain
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(24601)
    out: list[Case] = []
    n = 60
    x = g.normal(size=(n, 3))
    # Targets with a planted dependence: the third depends on the first two.
    t0 = x @ np.array([1.0, -0.5, 0.2]) + 0.1 * g.normal(size=n)
    t1 = np.sin(x[:, 0]) + 0.1 * g.normal(size=n)
    t2 = 0.8 * t0 - 0.6 * t1 + 0.05 * g.normal(size=n)
    y = np.column_stack([t0, t1, t2])
    q = g.normal(size=(9, 3))
    mor = MultiOutputRegressor(LinearRegression()).fit(x, y)
    out.append(
        case("regressor/independent", "regressor", [x, y, q], {"predict": mor.predict(q)}, rtol=1e-9, atol=1e-10)
    )
    for order in ([2, 0, 1], [0, 1, 2], [1, 2, 0]):
        chain = RegressorChain(LinearRegression(), order=order).fit(x, y)
        out.append(
            case(
                f"regressor/chain/{order}",
                "regressorChain",
                [x, y, q],
                {"predict": chain.predict(q)},
                rtol=1e-9,
                atol=1e-10,
                order=np.array(order, dtype=np.float64),
            )
        )
    counts = np.abs(g.normal(size=(n, 3)) * 2)
    labels = np.column_stack(
        [(counts[:, 0] > 1.5), np.digitize(counts[:, 1], [1.0, 2.5]), (counts[:, 2] > 2.0)]
    ).astype(np.float64)
    qc = np.abs(g.normal(size=(9, 3)) * 2)
    moc = MultiOutputClassifier(GaussianNB()).fit(counts, labels)
    proba = moc.predict_proba(qc)
    out.append(
        case(
            "classifier/independent",
            "classifier",
            [counts, labels, qc],
            {"predict": moc.predict(qc), "proba0": proba[0], "proba1": proba[1], "proba2": proba[2]},
            rtol=1e-9,
            atol=1e-10,
        )
    )
    return out
