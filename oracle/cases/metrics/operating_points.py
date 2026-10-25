"""Operating points against torchmetrics (fixed-rate metrics, nearest EER) and sklearn's ROC curve (the interpolated
EER by brentq on interp1d, Youden's index, minimum detection cost and cost curves from their definitions).
"""

from importlib.metadata import version

import numpy as np
import scipy
import sklearn
import sklearn.metrics as M
import torch
import torchmetrics.functional.classification as TC
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq

REFERENCE: dict[str, str] = {
    "torchmetrics": version("torchmetrics"),
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
}


# torchmetrics computes these curves in float32, so its results carry float32 rounding.
F32 = 1e-6


def eer_brentq(y: FloatArray, s: FloatArray) -> float:
    fpr, tpr, _ = M.roc_curve(y, s, drop_intermediate=False)
    curve = interp1d(fpr, tpr)

    def gap(x: float) -> float:
        return float(1.0 - x - curve(x))

    return float(brentq(gap, 0.0, 1.0, xtol=1e-14))


def min_dcf(y: FloatArray, s: FloatArray, prior: float, c_fp: float, c_fn: float) -> float:
    fpr, tpr, _ = M.roc_curve(y, s, drop_intermediate=False)
    dcf = prior * c_fn * (1 - tpr) + (1 - prior) * c_fp * fpr
    return float(dcf.min() / min(prior * c_fn, (1 - prior) * c_fp))


def cases() -> list[Case]:
    g = rng(56)
    n = 80
    y = (g.uniform(size=n) > 0.5).astype(np.float64)
    s = 1 / (1 + np.exp(-(2 * y - 1 + g.normal(size=n))))
    ty, ts = torch.tensor(y, dtype=torch.long), to_torch(s)
    out: list[Case] = []
    for m in [0.5, 0.8, 0.95]:
        r, _ = TC.binary_recall_at_fixed_precision(ts, ty, min_precision=m)
        p, _ = TC.binary_precision_at_fixed_recall(ts, ty, min_recall=m)
        se, _ = TC.binary_sensitivity_at_specificity(ts, ty, min_specificity=m)
        sp, _ = TC.binary_specificity_at_sensitivity(ts, ty, min_sensitivity=m)
        out += [
            case(
                f"recallAtFixedPrecision/{m}", "recallAtFixedPrecision", [y, s], np.asarray(float(r)), min=m, rtol=F32
            ),
            case(
                f"precisionAtFixedRecall/{m}", "precisionAtFixedRecall", [y, s], np.asarray(float(p)), min=m, rtol=F32
            ),
            case(
                f"sensitivityAtSpecificity/{m}",
                "sensitivityAtSpecificity",
                [y, s],
                np.asarray(float(se)),
                min=m,
                rtol=F32,
            ),
            case(
                f"specificityAtSensitivity/{m}",
                "specificityAtSensitivity",
                [y, s],
                np.asarray(float(sp)),
                min=m,
                rtol=F32,
            ),
        ]
    fpr, tpr, _ = M.roc_curve(y, s, drop_intermediate=False)
    fpr95 = float(fpr[np.argmax(tpr >= 0.95)])
    out += [
        case("fprAtTpr/0.95", "fprAtTpr", [y, s], np.asarray(fpr95), min=0.95),
        case(
            "equalErrorRate/nearest",
            "equalErrorRate",
            [y, s],
            np.asarray(float(TC.binary_eer(ts, ty))),
            method="nearest",
            rtol=F32,
        ),
        case("equalErrorRate/interpolated", "equalErrorRate", [y, s], np.asarray(eer_brentq(y, s)), rtol=1e-10),
        case("youdenIndex", "youdenIndex", [y, s], np.asarray(float((tpr - fpr).max()))),
        case("minimumExpectedCost", "minimumExpectedCost", [y, s], np.asarray(min_dcf(y, s, float(y.mean()), 1, 1))),
        case(
            "minimumExpectedCost/nist",
            "minimumExpectedCost",
            [y, s],
            np.asarray(min_dcf(y, s, 0.01, 1, 10)),
            prior=0.01,
            costFp=1,
            costFn=10,
        ),
    ]
    pc = np.linspace(0, 1, 11)
    nec = np.array([np.min((1 - tpr) * p + fpr * (1 - p)) for p in pc])
    out.append(case("costCurve", "costCurve", [y, s], nec, points=11))
    return out
