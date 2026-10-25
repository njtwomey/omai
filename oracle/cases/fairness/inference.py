"""The minimum detectable difference in a rate between two groups against statsmodels' NormalIndPower.

statsmodels' solve_power stops about 1e-7 from the root, so the reference solves its power function with scipy's brentq
at a tight tolerance instead.
"""

import numpy as np
import scipy.optimize
import statsmodels
from mdml_oracle.fixture import Case, case
from statsmodels.stats.power import NormalIndPower

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for n1, n2, alpha, power in [(100, 100, 0.05, 0.8), (400, 60, 0.05, 0.9), (35, 220, 0.01, 0.8)]:

        def gap(h: float, n1: int = n1, n2: int = n2, alpha: float = alpha, power: float = power) -> float:
            return float(NormalIndPower().power(effect_size=h, nobs1=n1, alpha=alpha, ratio=n2 / n1)) - power  # pyright: ignore[reportArgumentType]

        h = scipy.optimize.brentq(gap, 1e-10, 40, xtol=1e-14)
        out.append(
            case(
                f"minimumDetectableDifference/{n1}-{n2}-{alpha}-{power}",
                "effectSize",
                [],
                np.asarray(float(h)),
                rtol=1e-9,
                atol=1e-10,
                n1=n1,
                n2=n2,
                alpha=alpha,
                power=power,
            )
        )
    return out
