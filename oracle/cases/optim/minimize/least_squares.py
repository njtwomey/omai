"""Gauss-Newton and curve_fit against scipy.optimize: least_squares solutions of the standard residuals and curve_fit's
parameters and covariance (plain, weighted with relative and absolute sigma, and bounded)."""

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from omai_oracle.optim_problems import LS_T, LS_Y, RESIDUALS, exp_model, ls_summary

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["rosenbrock", "helical", "beale", "exponential"]:
        f, x0 = RESIDUALS[name]
        r = so.least_squares(f, x0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(
            case(f"gn/{name}", "gaussNewton", [x0, LS_T, LS_Y], ls_summary(r), rtol=1e-6, atol=1e-8, residuals=name)
        )
    sigma = 0.02 + 0.01 * LS_T
    fits: list[tuple[str, FloatArray | None, bool, tuple[list[float], list[float]] | None]] = [
        ("plain", None, False, None),
        ("sigma", sigma, False, None),
        ("absolute", sigma, True, None),
        ("bounded", None, False, ([0.0, 0.0, 0.5], [2.0, 5.0, 1.0])),
    ]
    for label, sg, absolute, bnds in fits:
        if bnds is None:
            popt, pcov = so.curve_fit(exp_model, LS_T, LS_Y, p0=[1.0, 1.0, 0.0], sigma=sg, absolute_sigma=absolute)
        else:
            popt, pcov = so.curve_fit(
                exp_model, LS_T, LS_Y, p0=[1.0, 1.0, 0.5], bounds=bnds, method="trf", ftol=1e-14, xtol=1e-14, gtol=1e-14
            )
        params: dict[str, object] = {"absoluteSigma": absolute}
        if bnds is not None:
            params["lower"] = bnds[0]
            params["upper"] = bnds[1]
        out.append(
            case(
                f"curveFit/{label}",
                "curveFit",
                [LS_T, LS_Y, np.zeros(1) if sg is None else sg],
                np.concatenate([np.asarray(popt, dtype=np.float64), np.asarray(pcov, dtype=np.float64).ravel()]),
                rtol=1e-5,
                atol=1e-8,
                hasSigma=sg is not None,
                **params,
            )
        )
    return out
