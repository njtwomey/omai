"""The trust region reflective method against scipy.optimize.least_squares(method='trf'): unbounded, bounded and with
robust losses."""

from typing import Literal

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import BOUNDED_RESIDUALS, LS_T, LS_Y, LS_Y_OUTLIERS, RESIDUALS, ls_summary

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Loss = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]

LOSSES: list[Loss] = ["soft_l1", "huber", "cauchy", "arctan"]


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["rosenbrock", "helical", "beale", "exponential"]:
        f, x0 = RESIDUALS[name]
        r = so.least_squares(f, x0, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(case(f"trf/{name}", "trf", [x0, LS_T, LS_Y], ls_summary(r), rtol=1e-7, atol=1e-9, residuals=name))
    for name, bounds in BOUNDED_RESIDUALS:
        f, x0 = RESIDUALS[name]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        start = np.clip(x0, lb, ub)
        r = so.least_squares(f, start, method="trf", bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(
            case(
                f"trf-bounded/{name}",
                "trf",
                [start, LS_T, LS_Y, lb, ub],
                np.concatenate([ls_summary(r), np.asarray(r.active_mask, dtype=np.float64)]),
                rtol=1e-6,
                atol=1e-8,
                residuals=name,
            )
        )
    for loss in LOSSES:
        f, x0 = RESIDUALS["outliers"]
        r = so.least_squares(f, x0, method="trf", loss=loss, f_scale=0.1, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(
            case(
                f"robust-trf/{loss}",
                "trf",
                [x0, LS_T, LS_Y_OUTLIERS],
                ls_summary(r),
                rtol=1e-6,
                atol=1e-8,
                residuals="outliers",
                loss=loss,
            )
        )
    return out
