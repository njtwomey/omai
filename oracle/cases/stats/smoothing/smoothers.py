"""Smoothers against statsmodels (lowess, KernelReg), pandas (rolling windows) and scipy (make_smoothing_spline)."""

import numpy as np
import pandas as pd
import scipy
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.interpolate import make_smoothing_spline
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess

REFERENCE: dict[str, str] = {
    "statsmodels": statsmodels.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
}


def cases() -> list[Case]:
    g = rng(77)
    x = g.uniform(0, 10, 60)
    y = np.sin(x) + 0.3 * g.normal(size=60)
    y[[5, 17, 40]] += 4.0
    out: list[Case] = []
    for frac, it, delta in [(2 / 3, 3, 0.0), (0.3, 0, 0.0), (0.4, 2, 0.5)]:
        fit = lowess(y, x, frac=frac, it=it, delta=delta)
        out.append(
            case(
                f"lowess/{frac:.2f}/{it}/{delta}",
                "lowess",
                [x, y],
                fit[:, 1],
                rtol=1e-10,
                atol=1e-12,
                frac=frac,
                it=it,
                delta=delta,
            )
        )
    at = np.linspace(0.5, 9.5, 7)
    for reg in ["lc", "ll"]:
        kr = KernelReg(y, x, var_type="c", reg_type=reg, bw=[0.8])  # pyright: ignore[reportArgumentType]
        mean, _ = kr.fit(at)
        out.append(
            case(
                f"kernelReg/{reg}",
                "nadarayaWatson" if reg == "lc" else "localLinear",
                [x, y, at],
                np.asarray(mean),
                rtol=1e-10,
                bandwidth=0.8,
            )
        )
    s = pd.Series(y)
    for w in [3, 4, 7]:
        out.append(
            case(
                f"runningMean/{w}",
                "runningMean",
                [y, y],
                np.asarray(s.rolling(w, center=True).mean(), dtype=float),
                window=w,
            )
        )
        out.append(
            case(
                f"runningMedian/{w}",
                "runningMedian",
                [y, y],
                np.asarray(s.rolling(w, center=True).median(), dtype=float),
                window=w,
            )
        )
        out.append(
            case(
                f"runningMean/trailing/{w}",
                "runningMeanTrailing",
                [y, y],
                np.asarray(s.rolling(w).mean(), dtype=float),
                window=w,
            )
        )
    xs = np.sort(x)
    ys = np.sin(xs) + 0.2 * g.normal(size=60)
    spline = make_smoothing_spline(xs, ys, lam=0.5)
    out.append(case("smoothingSpline", "smoothingSpline", [xs, ys], spline(at), rtol=1e-8, lam=0.5, at=at))
    return out
