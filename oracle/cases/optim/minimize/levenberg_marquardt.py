"""Levenberg-Marquardt against scipy.optimize.least_squares(method='lm') (MINPACK's lmder) on standard residuals."""

import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import LS_T, LS_Y, RESIDUALS, ls_summary

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["rosenbrock", "helical", "beale", "exponential"]:
        f, x0 = RESIDUALS[name]
        r = so.least_squares(f, x0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(
            case(
                f"lm/{name}",
                "levenbergMarquardt",
                [x0, LS_T, LS_Y],
                ls_summary(r),
                rtol=1e-7,
                atol=1e-9,
                residuals=name,
            )
        )
    return out
