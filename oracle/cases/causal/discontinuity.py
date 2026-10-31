"""Sharp regression discontinuity at a fixed bandwidth against statsmodels: a kernel-weighted polynomial fit on each
side of the cutoff (WLS with HC1 errors), the jump being the difference of the two intercepts and its variance the
sum of theirs."""

import numpy as np
import statsmodels
import statsmodels.api as sm
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def kernel(name: str, u: FloatArray) -> FloatArray:
    inside = np.abs(u) < 1
    if name == "uniform":
        return np.where(inside, 0.5, 0.0)
    if name == "epanechnikov":
        return np.where(inside, 0.75 * (1 - u**2), 0.0)
    return np.where(inside, 1 - np.abs(u), 0.0)


def side(
    x: FloatArray, y: FloatArray, cutoff: float, h: float, name: str, order: int, above: bool
) -> tuple[float, float]:
    v = x - cutoff
    w = kernel(name, v / h)
    keep = (v >= 0) == above
    keep &= w > 0
    design = np.column_stack([v[keep] ** j for j in range(order + 1)])
    fit = sm.WLS(y[keep], design, weights=w[keep].tolist()).fit(cov_type="HC1")  # pyright: ignore[reportUnknownMemberType]
    return float(fit.params[0]), float(fit.bse[0] ** 2)


def rd_case(name: str, seed: int, n: int, cutoff: float, h: float, kernel_name: str, order: int) -> Case:
    g = rng(seed)
    x = g.normal(loc=cutoff, scale=1.5, size=n)
    y = 1 + 0.8 * (x - cutoff) + 0.3 * (x - cutoff) ** 2 + 1.7 * (x >= cutoff) + 0.6 * g.normal(size=n)
    left, vl = side(x, y, cutoff, h, kernel_name, order, False)
    right, vr = side(x, y, cutoff, h, kernel_name, order, True)
    return case(
        name,
        "regressionDiscontinuity",
        [x, y],
        {"value": np.array(right - left), "se": np.array(float(np.sqrt(vl + vr)))},
        rtol=1e-8,
        atol=1e-10,
        cutoff=cutoff,
        bandwidth=h,
        kernel=kernel_name,
        order=order,
    )


def cases() -> list[Case]:
    return [
        rd_case("triangular/linear", 301, 1500, 0.0, 1.2, "triangular", 1),
        rd_case("uniform/linear", 302, 1500, 0.5, 1.0, "uniform", 1),
        rd_case("epanechnikov/quadratic", 303, 2000, -0.3, 1.8, "epanechnikov", 2),
        rd_case("triangular/quadratic", 304, 1200, 0.2, 1.5, "triangular", 2),
    ]
