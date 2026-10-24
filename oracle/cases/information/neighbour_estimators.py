"""k-nearest-neighbour estimators against NPEET and sklearn's mutual-information estimators."""

import importlib.metadata
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import sklearn
from npeet import entropy_estimators as ee  # pyright: ignore[reportMissingTypeStubs]
from sklearn.feature_selection._mutual_info import (  # pyright: ignore[reportPrivateUsage]
    _compute_mi_cc,  # pyright: ignore[reportPrivateUsage]
    _compute_mi_cd,  # pyright: ignore[reportPrivateUsage]
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

# NPEET annotates `base` as int through its defaults; any positive base works, and e gives nats.
NATS: Any = np.e

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "npeet": importlib.metadata.version("npeet")}

# NPEET adds noise of size 1e-10 to break ties; the estimates move by about that much, far inside these tolerances.
KNN_RTOL = 1e-7


def seeded(f: Callable[..., object], *args: object, **kwargs: object) -> float:
    """Call an NPEET estimator with numpy's global generator seeded (its jitter uses np.random)."""
    np.random.seed(0)  # noqa: NPY002
    return float(cast(float, f(*args, **kwargs)))


def cases() -> list[Case]:
    g = rng(420)
    out: list[Case] = []
    n = 300
    x1 = g.normal(size=(n, 1))
    x2 = g.normal(size=(n, 2))
    y1 = x1 + 0.5 * g.normal(size=(n, 1))
    y2 = np.c_[x2[:, :1] ** 2, x2[:, 1:] - x2[:, :1]] + 0.3 * g.normal(size=(n, 2))
    z1 = x1 + g.normal(size=(n, 1))
    for k in [3, 5]:
        out.append(
            case(
                f"kozachenkoLeonenkoEntropy/1d/k={k}",
                "kozachenkoLeonenkoEntropy",
                [x1],
                np.array(seeded(ee.entropy, x1, k=k, base=NATS)),
                rtol=KNN_RTOL,
                k=k,
            )
        )
        out.append(
            case(
                f"ksgMutualInformation/1d/k={k}",
                "ksgMutualInformation",
                [x1, y1],
                np.array(seeded(ee.mi, x1, y1, k=k, base=NATS)),
                rtol=KNN_RTOL,
                k=k,
            )
        )
    out.append(
        case(
            "kozachenkoLeonenkoEntropy/2d/base=2",
            "kozachenkoLeonenkoEntropy",
            [x2],
            np.array(seeded(ee.entropy, x2, k=4, base=2)),
            rtol=KNN_RTOL,
            k=4,
            base=2,
        )
    )
    out.append(
        case(
            "ksgMutualInformation/2d",
            "ksgMutualInformation",
            [x2, y2],
            np.array(seeded(ee.mi, x2, y2, k=3, base=NATS)),
            rtol=KNN_RTOL,
            k=3,
        )
    )
    # sklearn's KSG (mutual_info_regression without its scaling and jitter) on tie-free data.
    out.append(
        case(
            "ksgMutualInformation/sklearn",
            "ksgMutualInformation",
            [x1[:, 0], y1[:, 0]],
            np.array(_compute_mi_cc(x1[:, 0], y1[:, 0], 3)),
            rtol=1e-10,
            k=3,
        )
    )
    out.append(
        case(
            "ksgConditionalMutualInformation",
            "ksgConditionalMutualInformation",
            [x1, y1, z1],
            np.array(seeded(ee.cmi, x1, y1, z1, k=3, base=NATS)),
            rtol=KNN_RTOL,
            k=3,
        )
    )
    q2 = g.normal(size=(250, 2)) * 1.5 + 0.3
    out.append(
        case(
            "knnKlDivergence",
            "knnKlDivergence",
            [x2, q2],
            np.array(ee.kldiv(x2, q2, k=3, base=NATS)),
            rtol=1e-10,
            k=3,
        )
    )
    labels = g.integers(0, 3, size=n).astype(np.float64)
    c = labels + g.normal(size=n)
    out.append(
        case(
            "mixedMutualInformation",
            "mixedMutualInformation",
            [c, labels],
            np.array(_compute_mi_cd(c, labels, 3)),
            rtol=1e-10,
            k=3,
        )
    )
    return out
