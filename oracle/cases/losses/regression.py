"""Regression losses against torch.nn.functional and sklearn.metrics (float64)."""

from typing import Any

import numpy as np
import sklearn
import sklearn.metrics as M
import torch
import torch.nn.functional as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__, "scikit-learn": sklearn.__version__}


def reduced(terms: FloatArray, reduction: str) -> FloatArray:
    """Reduce elementwise terms as torch does."""
    if reduction == "mean":
        return np.asarray(terms.mean())
    if reduction == "sum":
        return np.asarray(terms.sum())
    return terms


def tweedie_terms(y: FloatArray, mu: FloatArray, power: float) -> FloatArray:
    """The Tweedie unit deviance of each element, from sklearn's mean deviance of one element."""
    flat = [M.mean_tweedie_deviance([a], [b], **tweedie(power)) for a, b in zip(y.ravel(), mu.ravel(), strict=True)]
    return np.asarray(flat, dtype=np.float64).reshape(y.shape)


def tweedie(power: float) -> dict[str, Any]:
    """Keyword arguments for sklearn's Tweedie deviance (its stubs type the power as an int)."""
    return {"power": power}


def cases() -> list[Case]:
    g = rng(41)
    x, y = g.normal(size=(4, 3)), g.normal(size=(4, 3))
    var = np.exp(g.normal(size=(4, 3)))
    counts = g.poisson(2.0, size=(4, 3)).astype(np.float64)
    w = g.uniform(0.2, 2.0, size=(4, 3))
    pos_y = np.where(counts > 0, counts, 0.0)
    mu = np.exp(0.5 * g.normal(size=(4, 3)))
    gamma_y = np.exp(g.normal(size=(4, 3)))
    out: list[Case] = []
    for reduction in ["mean", "sum", "none"]:
        r: dict[str, Any] = {"reduction": reduction}
        tx, ty = to_torch(x), to_torch(y)
        out += [
            case(f"mse/{reduction}", "mse", [x, y], to_numpy(F.mse_loss(tx, ty, **r)), **r),
            case(f"mae/{reduction}", "mae", [x, y], to_numpy(F.l1_loss(tx, ty, **r)), **r),
            case(f"huber/{reduction}", "huber", [x, y], to_numpy(F.huber_loss(tx, ty, delta=0.7, **r)), delta=0.7, **r),
            case(
                f"smoothL1/{reduction}",
                "smoothL1",
                [x, y],
                to_numpy(F.smooth_l1_loss(tx, ty, beta=0.6, **r)),
                beta=0.6,
                **r,
            ),
            case(
                f"poissonNll/{reduction}",
                "poissonNll",
                [x, counts],
                to_numpy(F.poisson_nll_loss(tx, to_torch(counts), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"poissonNll/rates/{reduction}",
                "poissonNll",
                [var, counts],
                to_numpy(F.poisson_nll_loss(to_torch(var), to_torch(counts), log_input=False, **r)),
                logInput=False,
                **r,
            ),
            case(
                f"gaussianNll/{reduction}",
                "gaussianNll",
                [x, y, var],
                to_numpy(F.gaussian_nll_loss(tx, ty, to_torch(var), full=True, **r)),
                full=True,
                **r,
            ),
            case(
                f"tweedieDeviance/1.5/{reduction}",
                "tweedieDeviance",
                [mu, pos_y],
                reduced(tweedie_terms(pos_y, mu, 1.5), reduction),
                power=1.5,
                **r,
            ),
        ]
    for power, target in [(0.0, y), (1.0, pos_y), (2.0, gamma_y), (3.0, gamma_y)]:
        out.append(
            case(
                f"tweedieDeviance/power={power}",
                "tweedieDeviance",
                [mu, target],
                np.asarray(M.mean_tweedie_deviance(target.ravel(), mu.ravel(), **tweedie(power))),
                power=power,
            )
        )
    out.append(
        case(
            "tweedieDeviance/logInput",
            "tweedieDeviance",
            [np.log(mu), pos_y],
            np.asarray(M.mean_tweedie_deviance(pos_y.ravel(), mu.ravel(), **tweedie(1.3))),
            power=1.3,
            logInput=True,
        )
    )
    # Pinball and expectile: the mean pinball loss is sklearn's; the expectile is the asymmetric squared error.
    for tau in [0.1, 0.5, 0.8]:
        out.append(
            case(
                f"pinball/tau={tau}",
                "pinball",
                [x.ravel(), y.ravel()],
                np.asarray(M.mean_pinball_loss(y.ravel(), x.ravel(), alpha=tau)),
                tau=tau,
            )
        )
        res = y - x
        out.append(
            case(
                f"expectile/tau={tau}",
                "expectile",
                [x, y],
                np.asarray((np.where(res < 0, 1 - tau, tau) * res**2).mean()),
                tau=tau,
            )
        )
    out.append(
        case(
            "logCosh",
            "logCosh",
            [x * 10, y],
            np.asarray(np.log(np.cosh(x * 10 - y)).mean()),
        )
    )
    out.append(
        case(
            "mse/weighted",
            "mse",
            [x, y, w],
            np.asarray(((x - y) ** 2 * w).sum() / w.sum()),
        )
    )
    return out
