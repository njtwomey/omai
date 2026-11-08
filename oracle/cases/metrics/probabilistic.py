"""Probabilistic classification metrics against sklearn (log loss, Brier, D^2, calibration curves), torchmetrics
(calibration errors) and the Murphy and classwise-ECE definitions written over sklearn's calibration curve.
"""

from importlib.metadata import version
from typing import Any, Literal

import numpy as np
import sklearn
import sklearn.metrics as M
import torch
import torchmetrics.functional.classification as TC
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_torch
from sklearn.calibration import calibration_curve

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "torchmetrics": version("torchmetrics")}

# torchmetrics bins in float32, so its calibration errors carry float32 rounding.
F32 = 1e-6

# sklearn types scale_by_half as a string; the boolean passes through a mapping instead.
HALF: dict[str, Any] = {"scale_by_half": True}


def murphy(y: FloatArray, p: FloatArray, n_bins: int) -> tuple[float, float, float]:
    """Reliability, resolution and uncertainty over sklearn's uniform calibration bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.searchsorted(bins[1:-1], p)
    obar = y.mean()
    rel = res = 0.0
    for b in np.unique(ids):
        m = ids == b
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2 / len(y)
        res += m.sum() * (y[m].mean() - obar) ** 2 / len(y)
    return float(rel), float(res), float(obar * (1 - obar))


def classwise_ece(y: FloatArray, probs: FloatArray, n_bins: int) -> float:
    total = 0.0
    for j in range(probs.shape[1]):
        ce = TC.binary_calibration_error(to_torch(probs[:, j]), torch.tensor((y == j).astype(int)), n_bins=n_bins)
        total += float(ce)
    return total / probs.shape[1]


def cases() -> list[Case]:
    g = rng(57)
    n = 80
    y = (g.uniform(size=n) > 0.5).astype(np.float64)
    p = 1 / (1 + np.exp(-(2 * y - 1 + 1.2 * g.normal(size=n))))
    w = g.uniform(0.2, 2.0, size=n)
    yk = g.integers(0, 3, size=n).astype(np.float64)
    logits = g.normal(size=(n, 3)) + 1.5 * np.eye(3)[yk.astype(int)]
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ty, tk = torch.tensor(y.astype(int)), torch.tensor(yk.astype(int))
    out: list[Case] = [
        case("logLoss/binary", "logLoss", [y, p], np.asarray(M.log_loss(y, p))),
        case("logLoss/weighted", "logLoss", [yk, probs, w], np.asarray(M.log_loss(yk, probs, sample_weight=w))),
        case(
            "logLoss/sum", "logLoss", [yk, probs], np.asarray(M.log_loss(yk, probs, normalize=False)), normalize=False
        ),
        case(
            "logLoss/clipped",
            "logLoss",
            [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
            np.asarray(M.log_loss([0, 1], [0.0, 1.0])),
        ),
        case("d2LogLossScore", "d2LogLossScore", [yk, probs], np.asarray(M.d2_log_loss_score(yk, probs))),
        case(
            "d2LogLossScore/weighted",
            "d2LogLossScore",
            [y, p, w],
            np.asarray(M.d2_log_loss_score(y, p, sample_weight=w)),
        ),
        case("brierScoreLoss/binary", "brierScoreLoss", [y, p], np.asarray(M.brier_score_loss(y, p))),
        case(
            "brierScoreLoss/weighted",
            "brierScoreLoss",
            [y, p, w],
            np.asarray(M.brier_score_loss(y, p, sample_weight=w)),
        ),
        case(
            "brierScoreLoss/posLabel0",
            "brierScoreLoss",
            [y, 1 - p],
            np.asarray(M.brier_score_loss(y, 1 - p, pos_label=0)),
            posLabel=0,
        ),
        case("brierScoreLoss/multiclass", "brierScoreLoss", [yk, probs], np.asarray(M.brier_score_loss(yk, probs))),
        case(
            "brierScoreLoss/multiclass-half",
            "brierScoreLoss",
            [yk, probs],
            np.asarray(M.brier_score_loss(yk, probs, **HALF)),
            scaleByHalf=True,
        ),
    ]
    for strategy in ["uniform", "quantile"]:
        pt, pp = calibration_curve(y, p, n_bins=6, strategy=strategy)
        out += [
            case(
                f"calibrationCurve/{strategy}/probTrue", "calibrationProbTrue", [y, p], pt, nBins=6, strategy=strategy
            ),
            case(
                f"calibrationCurve/{strategy}/probPred", "calibrationProbPred", [y, p], pp, nBins=6, strategy=strategy
            ),
        ]
    rel, res, unc = murphy(y, p, 10)
    out += [
        case("brierDecomposition/reliability", "brierReliability", [y, p], np.asarray(rel)),
        case("brierDecomposition/resolution", "brierResolution", [y, p], np.asarray(res)),
        case("brierDecomposition/uncertainty", "brierUncertainty", [y, p], np.asarray(unc)),
    ]
    norms: list[Literal["l1", "l2", "max"]] = ["l1", "l2", "max"]
    for norm in norms:
        name = "maximumCalibrationError" if norm == "max" else "expectedCalibrationError"
        params: dict[str, Any] = {} if norm == "max" else {"norm": norm}
        out += [
            case(
                f"{name}/binary-{norm}",
                name,
                [y, p],
                np.asarray(float(TC.binary_calibration_error(to_torch(p), ty, n_bins=10, norm=norm))),
                nBins=10,
                rtol=F32,
                **params,
            ),
            case(
                f"{name}/multiclass-{norm}",
                name,
                [yk, probs],
                np.asarray(
                    float(TC.multiclass_calibration_error(to_torch(probs), tk, num_classes=3, n_bins=15, norm=norm))
                ),
                rtol=F32,
                **params,
            ),
        ]
    out.append(
        case(
            "classwiseCalibrationError",
            "classwiseCalibrationError",
            [yk, probs],
            np.asarray(classwise_ece(yk, probs, 10)),
            nBins=10,
            rtol=F32,
        )
    )
    out.append(
        case(
            "expectedCalibrationError/quantile",
            "expectedCalibrationError",
            [y, p],
            np.asarray(adaptive_ece(y, p, 8)),
            nBins=8,
            strategy="quantile",
        )
    )
    return out


def adaptive_ece(y: FloatArray, p: FloatArray, n_bins: int) -> float:
    """ECE over equal-mass bins (sklearn calibration_curve's quantile edges)."""
    edges = np.percentile(p, np.linspace(0, 1, n_bins + 1) * 100)
    ids = np.searchsorted(edges[1:-1], p)
    return float(sum((ids == b).sum() / len(y) * abs(y[ids == b].mean() - p[ids == b].mean()) for b in np.unique(ids)))
