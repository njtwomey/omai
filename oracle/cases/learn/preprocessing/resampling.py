"""Resampling against imbalanced-learn: Tomek links and NearMiss (1, 2, 3) kept indices, the borderline-SMOTE danger
set, and the class counts after SMOTE, ADASYN and random sampling (the random draws themselves are tested as
properties, since imbalanced-learn's generator cannot be reproduced)."""

from collections import Counter

import imblearn
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler, TomekLinks
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.neighbors import NearestNeighbors

REFERENCE: dict[str, str] = {"imbalanced-learn": imblearn.__version__}


def counts(y: FloatArray) -> FloatArray:
    """Class counts in label order."""
    c = Counter(float(v) for v in y)
    return np.array([c[k] for k in sorted(c)], dtype=np.float64)


def cases() -> list[Case]:
    g = rng(4335)
    x0 = g.normal(size=(40, 2))
    x1 = g.normal(size=(12, 2)) * 0.8 + np.array([1.2, 1.0])
    x2 = g.normal(size=(8, 2)) * 0.5 + np.array([-1.5, 1.5])
    x = np.vstack([x0, x1, x2])
    y = np.array([0.0] * 40 + [1.0] * 12 + [2.0] * 8)
    out: list[Case] = []
    tl = TomekLinks()
    tl.fit_resample(x, y)
    out.append(case("tomek", "tomek", [x, y], {"indices": tl.sample_indices_.astype(np.int32)}))
    for version in [1, 2, 3]:
        nm = NearMiss(version=version, n_neighbors=3, n_neighbors_ver3=3)
        nm.fit_resample(x, y)
        out.append(
            case(
                f"nearMiss/{version}",
                "nearMiss",
                [x, y],
                {"indices": np.sort(nm.sample_indices_).astype(np.int32)},
                version=version,
            )
        )
    bs = BorderlineSMOTE(m_neighbors=8, k_neighbors=3)
    nn = NearestNeighbors(n_neighbors=9).fit(x)
    for label in [1.0, 2.0]:
        cls = np.flatnonzero(y == label)
        danger = bs._in_danger_noise(nn, x[cls], label, y, kind="danger")  # pyright: ignore[reportPrivateUsage]
        out.append(
            case(f"danger/{label}", "danger", [x, y], {"indices": cls[danger].astype(np.int32)}, label=label, m=8)
        )
    for name, sampler in [
        ("smote", SMOTE(k_neighbors=3, random_state=0)),
        ("adasyn", ADASYN(n_neighbors=3, random_state=0)),
        ("borderline", BorderlineSMOTE(m_neighbors=8, k_neighbors=3, random_state=0)),
        ("over", RandomOverSampler(random_state=0)),
        ("under", RandomUnderSampler(random_state=0)),
    ]:
        resampled = sampler.fit_resample(x, y)
        out.append(case(f"counts/{name}", "counts", [x, y], {"counts": counts(np.asarray(resampled[1]))}, sampler=name))
    return out
