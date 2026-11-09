"""The baselines against sklearn.dummy: DummyClassifier (prior, most frequent, constant; weighted and unweighted
class frequencies) and DummyRegressor (mean, median, quantile, constant; weighted mean), their predictions and
probabilities."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.dummy import DummyClassifier, DummyRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(5)
    x = g.normal(size=(40, 3))
    held = g.normal(size=(6, 3))
    y_class = np.asarray(g.choice([0.0, 1.0, 3.0], size=40, p=[0.2, 0.5, 0.3]), dtype=np.float64)
    weights = g.uniform(0.2, 2.0, size=40)
    y_reg = g.normal(loc=4.0, scale=2.0, size=40) ** 2
    out: list[Case] = []
    for strategy, constant in [("prior", None), ("most_frequent", None), ("constant", 3)]:
        for weighted in [False, True]:
            w = weights if weighted else None
            model = DummyClassifier(strategy=strategy, constant=constant).fit(x, y_class, sample_weight=w)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"classifier/{strategy}/{'weighted' if weighted else 'plain'}",
                    "classifier",
                    [x, y_class, held, weights],
                    {
                        "predict": model.predict(held).astype(np.float64),
                        "predictProba": model.predict_proba(held),
                        "prior": model.class_prior_,
                    },
                    rtol=1e-12,
                    atol=1e-12,
                    strategy=strategy,
                    constant=-1 if constant is None else constant,
                    weighted=weighted,
                )
            )
    for strategy, quantile, constant in [
        ("mean", None, None),
        ("median", None, None),
        ("quantile", 0.3, None),
        ("quantile", 0.9, None),
        ("constant", None, 2.5),
    ]:
        for weighted in [False, True]:
            if weighted and strategy in ("median", "quantile"):
                continue  # scikit-learn's weighted percentile convention is not the one omai documents
            label = strategy if quantile is None else f"{strategy}/{quantile}"
            w = weights if weighted else None
            model = DummyRegressor(strategy=strategy, quantile=quantile, constant=constant).fit(
                x, y_reg, sample_weight=w
            )  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"regressor/{label}/{'weighted' if weighted else 'plain'}",
                    "regressor",
                    [x, y_reg, held, weights],
                    {"predict": model.predict(held)},
                    rtol=1e-12,
                    atol=1e-12,
                    strategy=strategy,
                    quantile=-1.0 if quantile is None else quantile,
                    constant=-1.0 if constant is None else constant,
                    weighted=weighted,
                )
            )
    return out
