"""Dataset summaries and table encoding against numpy and scikit-learn."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.preprocessing import OneHotEncoder

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}


def summary(x: FloatArray) -> FloatArray:
    """Per column: count, missing, mean, std (ddof 1), min, quartiles and max, ignoring NaN (pandas describe)."""
    rows: list[list[float]] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        present = col[~np.isnan(col)]
        rows.append(
            [
                float(present.size),
                float(col.size - present.size),
                float(np.mean(present)),
                float(np.std(present, ddof=1)),
                float(np.min(present)),
                float(np.quantile(present, 0.25)),
                float(np.quantile(present, 0.5)),
                float(np.quantile(present, 0.75)),
                float(np.max(present)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(0)
    for n, d in [(7, 2), (20, 3)]:
        x = g.normal(size=(n, d))
        x[1, 0] = np.nan
        out.append(case(f"describe/{n}x{d}", "describe", [x], summary(x), rtol=1e-12, atol=1e-12))
    colours = ["red", "blue", "green", "blue", "red", "red"]
    enc = OneHotEncoder(sparse_output=False).fit_transform(np.asarray(colours).reshape(-1, 1))
    out.append(case("tableToXy/onehot", "tableToXy", [], np.asarray(enc), colours=",".join(colours)))
    drop = OneHotEncoder(sparse_output=False, drop="first").fit_transform(np.asarray(colours).reshape(-1, 1))
    out.append(case("tableToXy/drop-first", "tableToXy", [], np.asarray(drop), colours=",".join(colours), drop=1))
    return out
