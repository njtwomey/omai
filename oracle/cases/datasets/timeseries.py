"""ARMA autocovariances against statsmodels.tsa.arima_process."""

import numpy as np
import statsmodels
from mdml_oracle.fixture import Case, case
from statsmodels.tsa.arima_process import arma_acovf

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for ar, ma, sigma in [
        ([0.5], [], 1.0),
        ([0.6, -0.2], [0.4], 2.0),
        ([], [0.3, 0.2], 0.5),
        ([1.2, -0.5], [-0.3], 1.0),
    ]:
        # statsmodels writes the lag polynomials with a leading one and the AR signs flipped.
        # statsmodels annotates sigma2 as int, but it is the (real) innovation variance.
        variance: int = sigma**2  # pyright: ignore[reportAssignmentType]
        acov = arma_acovf(np.r_[1, -np.asarray(ar)], np.r_[1, np.asarray(ma)], nobs=10, sigma2=variance)
        name = f"acvf/ar{ar}/ma{ma}".replace(" ", "")
        out.append(
            case(
                name,
                "armaProcess",
                [np.asarray(ar, dtype=np.float64), np.asarray(ma, dtype=np.float64)],
                np.asarray(acov),
                rtol=1e-8,
                sigma=sigma,
            )
        )
    return out
