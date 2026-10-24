"""Cross-validation splitters against sklearn.model_selection: the exact train and test indices of every fold."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

type Folds = Iterable[tuple[npt.ArrayLike, npt.ArrayLike]]


def folds_output(folds: Folds) -> dict[str, object]:
    """Concatenated sorted train and test indices with the fold sizes."""
    train: list[int] = []
    test: list[int] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    for tr_like, te_like in folds:
        tr = np.asarray(tr_like, dtype=np.int64)
        te = np.asarray(te_like, dtype=np.int64)
        train += sorted(int(i) for i in tr)
        test += sorted(int(i) for i in te)
        train_sizes.append(len(tr))
        test_sizes.append(len(te))
    return {
        "train": np.array(train, dtype=np.int32),
        "test": np.array(test, dtype=np.int32),
        "trainSizes": np.array(train_sizes, dtype=np.int32),
        "testSizes": np.array(test_sizes, dtype=np.int32),
    }


def cases() -> list[Case]:
    g = rng(4331)
    n = 23
    x = np.zeros((n, 1))
    y = np.array([2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 0, 2, 1, 2, 2, 0, 2, 2, 1], dtype=np.float64)
    groups = np.array([5, 5, 5, 1, 1, 9, 9, 9, 9, 3, 3, 3, 3, 3, 7, 7, 2, 2, 2, 2, 2, 2, 8], dtype=np.float64)
    out: list[Case] = []
    for k in [2, 3, 5]:
        out.append(case(f"kFold/{k}", "kFold", [y], folds_output(KFold(k).split(x)), nSplits=k))
        out.append(
            case(
                f"stratifiedKFold/{k}", "stratifiedKFold", [y], folds_output(StratifiedKFold(k).split(x, y)), nSplits=k
            )
        )
        out.append(
            case(
                f"groupKFold/{k}", "groupKFold", [y, groups], folds_output(GroupKFold(k).split(x, y, groups)), nSplits=k
            )
        )
        out.append(
            case(
                f"stratifiedGroupKFold/{k}",
                "stratifiedGroupKFold",
                [y, groups],
                folds_output(StratifiedGroupKFold(k).split(x, y, groups)),
                nSplits=k,
            )
        )
    # Groups of random sizes, including ties.
    g2 = g.integers(0, 9, size=40).astype(np.float64)
    y2 = g.integers(0, 3, size=40).astype(np.float64)
    x2 = np.zeros((40, 1))
    out.append(
        case("groupKFold/random", "groupKFold", [y2, g2], folds_output(GroupKFold(4).split(x2, y2, g2)), nSplits=4)
    )
    out.append(
        case(
            "stratifiedGroupKFold/random",
            "stratifiedGroupKFold",
            [y2, g2],
            folds_output(StratifiedGroupKFold(4).split(x2, y2, g2)),
            nSplits=4,
        )
    )
    out.append(
        case(
            "stratifiedKFold/random", "stratifiedKFold", [y2], folds_output(StratifiedKFold(4).split(x2, y2)), nSplits=4
        )
    )
    out.append(case("leaveOneOut", "leaveOneOut", [y[:6]], folds_output(LeaveOneOut().split(x[:6]))))
    out.append(case("leavePOut/2", "leavePOut", [y[:6]], folds_output(LeavePOut(2).split(x[:6])), p=2))
    out.append(
        case("leaveOneGroupOut", "leaveOneGroupOut", [y, groups], folds_output(LeaveOneGroupOut().split(x, y, groups)))
    )
    out.append(
        case(
            "leavePGroupsOut/2",
            "leavePGroupsOut",
            [y, groups],
            folds_output(LeavePGroupsOut(2).split(x, y, groups)),
            p=2,
        )
    )
    test_fold = np.array([0, 1, -1, 2, 1, 0, -1, 2, 2, 0], dtype=np.int32)
    out.append(
        case("predefinedSplit", "predefinedSplit", [test_fold], folds_output(PredefinedSplit(test_fold).split()))
    )
    for k, opts in [
        (3, {}),
        (4, {"max_train_size": 5}),
        (3, {"test_size": 4}),
        (3, {"gap": 2}),
        (2, {"test_size": 3, "gap": 1, "max_train_size": 6}),
    ]:
        label = "-".join(f"{a}={b}" for a, b in opts.items())
        params = {
            "maxTrainSize": opts.get("max_train_size", -1),
            "testSize": opts.get("test_size", -1),
            "gap": opts.get("gap", 0),
        }
        out.append(
            case(
                f"timeSeriesSplit/{k}/{label}",
                "timeSeriesSplit",
                [y],
                folds_output(TimeSeriesSplit(k, **opts).split(x)),
                nSplits=k,
                **params,
            )
        )
    idx = np.arange(n)
    for size in [0.25, 5]:
        tr, te = train_test_split(idx, test_size=size, shuffle=False)
        out.append(case(f"holdout/{size}", "holdout", [y], folds_output([(tr, te)]), testSize=size))
    tr, te = train_test_split(idx, test_size=0.3, train_size=0.5, shuffle=False)
    out.append(case("holdout/both", "holdout", [y], folds_output([(tr, te)]), testSize=0.3, trainSize=0.5))
    return out
