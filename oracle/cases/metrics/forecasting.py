"""Forecast metrics against torchmetrics (sMAPE, WMAPE) and properscoring (CRPS), with the scaled errors (MASE,
RMSSE, MSIS) as defined by Hyndman and Koehler and the M4 and M5 competitions, the interval score of Gneiting and
Raftery, quantile losses and Theil's U2 written over numpy.
"""

from importlib.metadata import version

import numpy as np
import properscoring as ps
import torchmetrics.functional.regression as TR
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torchmetrics": version("torchmetrics"), "properscoring": "0.1"}


def naive_scale(train: FloatArray, m: int, p: int) -> float:
    return float(np.mean(np.abs(train[m:] - train[:-m]) ** p))


def interval_scores(lo: FloatArray, hi: FloatArray, y: FloatArray, alpha: float) -> FloatArray:
    return (hi - lo) + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)


def pinball(y: FloatArray, q: FloatArray, tau: float) -> FloatArray:
    r = y - q
    return np.maximum(tau * r, (tau - 1) * r)


def cases() -> list[Case]:
    g = rng(63)
    train = np.cumsum(g.normal(size=30)) + 10 + 3 * np.sin(np.arange(30) * np.pi / 2)
    y = np.cumsum(g.normal(size=8)) + train[-1]
    p = y + 0.7 * g.normal(size=8)
    sigma = np.exp(0.3 * g.normal(size=8))
    lo, hi = p - 1.5 * sigma, p + 1.5 * sigma
    ens = p[:, None] + sigma[:, None] * g.normal(size=(8, 15))
    taus = [0.1, 0.5, 0.9]
    qf = np.stack([p + s * sigma for s in [-1.28, 0.0, 1.28]], axis=1)
    out: list[Case] = [
        case(
            "symmetricMeanAbsolutePercentageError",
            "symmetricMeanAbsolutePercentageError",
            [y, p],
            to_numpy(TR.symmetric_mean_absolute_percentage_error(to_torch(p), to_torch(y))),
        ),
        case(
            "weightedMeanAbsolutePercentageError",
            "weightedMeanAbsolutePercentageError",
            [y, p],
            to_numpy(TR.weighted_mean_absolute_percentage_error(to_torch(p), to_torch(y))),
        ),
        case(
            "meanAbsoluteScaledError",
            "meanAbsoluteScaledError",
            [y, p, train],
            np.asarray(np.mean(np.abs(y - p)) / naive_scale(train, 1, 1)),
        ),
        case(
            "meanAbsoluteScaledError/seasonal",
            "meanAbsoluteScaledError",
            [y, p, train],
            np.asarray(np.mean(np.abs(y - p)) / naive_scale(train, 4, 1)),
            seasonality=4,
        ),
        case(
            "rootMeanSquaredScaledError",
            "rootMeanSquaredScaledError",
            [y, p, train],
            np.asarray(np.sqrt(np.mean((y - p) ** 2) / naive_scale(train, 1, 2))),
        ),
        case(
            "intervalScore", "intervalScore", [lo, hi, y], np.asarray(interval_scores(lo, hi, y, 0.2).mean()), alpha=0.2
        ),
        case(
            "meanScaledIntervalScore",
            "meanScaledIntervalScore",
            [lo, hi, y, train],
            np.asarray(interval_scores(lo, hi, y, 0.05).mean() / naive_scale(train, 4, 1)),
            seasonality=4,
        ),
        case("intervalCoverage", "intervalCoverage", [lo, hi, y], np.asarray(float(np.mean((y >= lo) & (y <= hi))))),
        case(
            "quantileLoss",
            "quantileLoss",
            [y, qf],
            np.asarray(np.mean([pinball(y, qf[:, j], t).mean() for j, t in enumerate(taus)])),
            quantiles=taus,
        ),
        case(
            "quantileLoss/weighted",
            "quantileLoss",
            [y, qf],
            np.asarray(np.mean([2 * pinball(y, qf[:, j], t).sum() / np.abs(y).sum() for j, t in enumerate(taus)])),
            quantiles=taus,
            normalize="weighted",
        ),
        case(
            "crpsEnsemble",
            "crpsEnsemble",
            [y, ens],
            np.asarray(np.mean(np.asarray(ps.crps_ensemble(y, ens), dtype=np.float64))),
        ),
        case(
            "crpsNormal",
            "crpsNormal",
            [y, p, sigma],
            np.asarray(np.mean(np.asarray(ps.crps_gaussian(y, p, sigma), dtype=np.float64))),
        ),
        case(
            "theilsU2",
            "theilsU2",
            [y, p],
            np.asarray(np.sqrt(np.sum(((p[1:] - y[1:]) / y[:-1]) ** 2) / np.sum(((y[1:] - y[:-1]) / y[:-1]) ** 2))),
        ),
    ]
    fair = np.mean(
        [
            np.mean(np.abs(e - v)) - np.abs(e[:, None] - e[None, :]).sum() / (2 * 15 * 14)
            for e, v in zip(ens, y, strict=True)
        ]
    )
    out.append(case("crpsEnsemble/fair", "crpsEnsemble", [y, ens], np.asarray(fair), fair=True))
    return out
