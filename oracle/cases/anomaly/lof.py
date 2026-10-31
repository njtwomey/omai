"""Local outlier factors against scikit-learn: the training factors, the novelty scores of new points, and the
decision function and labels at a contamination, for the Euclidean and Manhattan metrics."""

from importlib.metadata import version

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.neighbors import LocalOutlierFactor

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-10}


def cases() -> list[Case]:
    g = rng(29)
    x = np.concatenate([g.normal(0, 1, (40, 2)), g.uniform(-6, 6, (5, 2))])
    q = g.uniform(-4, 4, (6, 2))
    out: list[Case] = []
    for metric in ["euclidean", "manhattan"]:
        for k in [5, 20]:
            lof = LocalOutlierFactor(n_neighbors=k, metric=metric).fit(x)
            out.append(
                case(
                    f"{metric}/k{k}/train",
                    "train",
                    [x],
                    {"scores": -np.asarray(lof.negative_outlier_factor_)},
                    **TOL,
                    k=k,
                    metric=metric,
                )
            )
            nov = LocalOutlierFactor(n_neighbors=k, metric=metric, novelty=True).fit(x)
            out.append(
                case(
                    f"{metric}/k{k}/novel",
                    "novel",
                    [x, q],
                    {"scores": -np.asarray(nov.score_samples(q))},
                    **TOL,
                    k=k,
                    metric=metric,
                )
            )
    for contamination in [0.1, 0.2]:
        nov = LocalOutlierFactor(
            n_neighbors=10,
            contamination=contamination,  # pyright: ignore[reportArgumentType]
            novelty=True,
        ).fit(x)
        out.append(
            case(
                f"contamination{contamination}",
                "contamination",
                [x, q],
                {
                    "decision": np.asarray(nov.decision_function(q)),
                    "predict": np.asarray(nov.predict(q), dtype=np.float64),
                    "threshold": np.asarray(float(-nov.offset_)),
                },
                **TOL,
                contamination=contamination,
            )
        )
    nov = LocalOutlierFactor(n_neighbors=10, novelty=True).fit(x)
    out.append(
        case(
            "auto",
            "auto",
            [x, q],
            {"decision": np.asarray(nov.decision_function(q)), "predict": np.asarray(nov.predict(q), dtype=np.float64)},
            **TOL,
        )
    )
    return out
