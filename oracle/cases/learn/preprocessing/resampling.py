"""Resampling against imbalanced-learn: Tomek links and NearMiss (1, 2, 3) kept indices, the borderline-SMOTE danger
set, and the class counts after SMOTE, ADASYN and random sampling (the random draws themselves are tested as
properties, since imbalanced-learn's generator cannot be reproduced). The edited nearest neighbours family (ENN,
repeated ENN, AllKNN), the neighbourhood cleaning rule, the condensing rules (condensed nearest neighbour and one-sided
selection, started from chosen samples through a scripted random state) and instance hardness (unshuffled folds, naive
Bayes) are deterministic, so their kept indices are fixtures; so is the value difference metric behind SMOTEN. The
counts after SMOTEN, SMOTENC, SVM-SMOTE and ClusterCentroids are checked too."""

from collections import Counter

import imblearn
import imblearn.under_sampling._prototype_selection._instance_hardness_threshold as hardness_module  # pyright: ignore[reportPrivateUsage]
import numpy as np
import numpy.typing as npt
from imblearn.metrics.pairwise import ValueDifferenceMetric
from imblearn.over_sampling import ADASYN, SMOTE, SMOTEN, SMOTENC, SVMSMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from omai_oracle.fixture import Case, case, encode_params
from omai_oracle.helpers import FloatArray, rng
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors

REFERENCE: dict[str, str] = {"imbalanced-learn": imblearn.__version__}


def counts(y: FloatArray) -> FloatArray:
    """Class counts in label order."""
    c = Counter(float(v) for v in y)
    return np.array([c[k] for k in sorted(c)], dtype=np.float64)


class ScriptedState(np.random.RandomState):
    """A random state whose `randint` replays chosen positions, so the condensing rules start from known samples."""

    def __init__(self, picks: list[int]) -> None:
        super().__init__(0)
        self.picks = list(picks)

    def randint(  # pyright: ignore[reportIncompatibleMethodOverride, reportImplicitOverride]
        self, low: int, high: int | None = None, size: int | None = None, dtype: type = int
    ) -> npt.NDArray[np.int64]:
        return np.array([self.picks.pop(0) for _ in range(size or 1)], dtype=np.int64)


class UnshuffledFolds(StratifiedKFold):
    """Stratified folds that ignore `shuffle`, so instance hardness is deterministic."""

    def __init__(self, n_splits: int = 5, *, shuffle: bool = False, random_state: object = None) -> None:
        super().__init__(n_splits=n_splits, shuffle=False)


def indices_case(
    name: str, fn: str, x: FloatArray, y: npt.NDArray[np.generic], sampler: object, **params: object
) -> Case:
    """The kept indices of a cleaning sampler (in its own order), with the resampled rows."""
    resampled = sampler.fit_resample(x, y)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    kept = np.asarray(sampler.sample_indices_, dtype=np.int32)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert np.array_equal(np.asarray(resampled[0]), x[kept])  # pyright: ignore[reportUnknownArgumentType]
    made = case(name, fn, [x, y], {"indices": kept})
    made.params.update(encode_params(params))
    return made


def cleaning_cases(x: FloatArray, y: FloatArray) -> list[Case]:
    """The deterministic cleaning and condensing rules on class-ordered and shuffled copies of the data."""
    g = rng(77)
    order = g.permutation(len(y))
    xs, ys = x[order], y[order]
    out: list[Case] = []
    for kind in ["all", "mode"]:
        out.append(
            indices_case(f"enn/{kind}", "enn", xs, ys, EditedNearestNeighbours(n_neighbors=3, kind_sel=kind), kind=kind)
        )
    out.append(indices_case("renn/sorted", "renn", x, y, RepeatedEditedNearestNeighbours(n_neighbors=3)))
    out.append(
        indices_case(
            "renn/mode", "renn", x, y, RepeatedEditedNearestNeighbours(n_neighbors=3, kind_sel="mode"), kind="mode"
        )
    )
    for allow in [False, True]:
        out.append(
            indices_case(
                f"allknn/{allow}", "allknn", x, y, AllKNN(n_neighbors=3, allow_minority=allow), allow=float(allow)
            )
        )
    out.append(indices_case("allknn/shuffled", "allknn", xs, ys, AllKNN(n_neighbors=4), n=4.0))
    out.append(indices_case("ncr", "ncr", xs, ys, NeighbourhoodCleaningRule(n_neighbors=3)))
    out.append(
        indices_case(
            "ncr/threshold",
            "ncr",
            x,
            y,
            NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.1),
            threshold=0.1,
        )
    )
    out.append(
        indices_case(
            "cnn", "cnn", x, y, CondensedNearestNeighbour(random_state=ScriptedState([7, 2])), seed0=7.0, seed1=2.0
        )
    )
    out.append(
        indices_case(
            "cnn/k3",
            "cnn",
            x,
            y,
            CondensedNearestNeighbour(random_state=ScriptedState([30, 5]), n_neighbors=3),
            seed0=30.0,
            seed1=5.0,
            k=3.0,
        )
    )
    out.append(
        indices_case("oss", "oss", x, y, OneSidedSelection(random_state=ScriptedState([7, 2])), seed0=7.0, seed1=2.0)
    )
    out.append(
        indices_case(
            "oss/k3",
            "oss",
            xs,
            ys,
            OneSidedSelection(random_state=ScriptedState([3, 4]), n_neighbors=3),
            seed0=3.0,
            seed1=4.0,
            k=3.0,
        )
    )
    original = hardness_module.StratifiedKFold
    hardness_module.StratifiedKFold = UnshuffledFolds
    try:
        out.append(
            indices_case(
                "iht",
                "iht",
                x,
                y.astype(np.int64),
                InstanceHardnessThreshold(estimator=GaussianNB(), cv=3, random_state=0),
            )
        )
    finally:
        hardness_module.StratifiedKFold = original
    return out


def categorical_cases() -> list[Case]:
    """The value difference metric, and the class counts after the categorical, mixed and SVM variants."""
    g = rng(51)
    xc = g.integers(0, 4, size=(45, 3)).astype(np.float64)
    yc = np.array([0.0] * 25 + [1.0] * 12 + [2.0] * 8)
    xc[:25, 0] = np.where(g.random(25) < 0.7, 0.0, xc[:25, 0])
    xc[25:37, 1] = np.where(g.random(12) < 0.7, 2.0, xc[25:37, 1])
    xc[37:, 2] = np.where(g.random(8) < 0.7, 3.0, xc[37:, 2])
    out: list[Case] = []
    vdm = ValueDifferenceMetric(n_categories="auto").fit(xc.astype(np.int32), yc.astype(np.int32))
    out.append(
        case("vdm", "vdm", [xc, yc, xc[:12]], {"distance": vdm.pairwise(xc[:12].astype(np.int32), xc.astype(np.int32))})
    )
    out.append(
        case(
            "counts/smoten",
            "counts",
            [xc, yc],
            {"counts": counts(np.asarray(SMOTEN(k_neighbors=3, random_state=0).fit_resample(xc, yc)[1]))},
            sampler="smoten",
        )
    )
    xm = np.column_stack([g.normal(size=45), g.integers(0, 3, size=45), g.normal(size=45) + yc]).astype(np.float64)
    out.append(
        case(
            "counts/smotenc",
            "counts",
            [xm, yc],
            {"counts": counts(np.asarray(SMOTENC([1], k_neighbors=3, random_state=0).fit_resample(xm, yc)[1]))},
            sampler="smotenc",
        )
    )
    return out


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
    centroids = ClusterCentroids(random_state=0).fit_resample(x, y)
    out.append(
        case("counts/centroids", "counts", [x, y], {"counts": counts(np.asarray(centroids[1]))}, sampler="centroids")
    )
    svm = SVMSMOTE(k_neighbors=3, m_neighbors=8, random_state=0).fit_resample(x, y)
    out.append(case("counts/svmsmote", "counts", [x, y], {"counts": counts(np.asarray(svm[1]))}, sampler="svmsmote"))
    return [*out, *cleaning_cases(x, y), *categorical_cases()]
