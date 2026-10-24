"""Sliding-window building blocks of the matrix profile against STUMPY: rolling statistics, constancy, sliding dot
products, z-normalisation and MASS distance profiles (z-normalised and non-normalised), with exclusion zones."""

from importlib.metadata import version

import numpy as np
import stumpy
from stumpy import core
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-10}
# A query taken from the series has an exact match where the distance is sqrt(2m(1 - rho)) with rho within rounding
# of 1: FFT rounding of 1e-14 in rho becomes 1e-6 in the distance, so near-zero distances get atol 1e-6.
SELF = {"rtol": 1e-8, "atol": 1e-6}


def series(n: int, seed: int) -> FloatArray:
    g = rng(seed)
    t = np.linspace(0, 8 * np.pi, n)
    return np.sin(t) + 0.4 * np.sin(2.7 * t) + 0.2 * g.normal(size=n)


def cases() -> list[Case]:
    out: list[Case] = []
    t = series(80, 501)
    t[30:38] = 0.75  # a constant stretch
    for m in [3, 8]:
        mean, std = core.compute_mean_std(t, m)
        out.append(case(f"rollingMeanStd/{m}", "rollingMeanStd", [t], np.stack([mean, std]), **TOL, m=m))
        out.append(
            case(f"rollingIsConstant/{m}", "rollingIsConstant", [t], core._rolling_isconstant(t, m).astype(float), m=m)  # pyright: ignore[reportPrivateUsage]
        )
    tn = t.copy()
    tn[50] = np.nan
    # Windows with a NaN: means are infinite (STUMPY's marker); their deviations are unused, so only means are compared.
    mean, _ = core.compute_mean_std(tn, 6)
    out.append(case("rollingMean/nan", "rollingMean", [tn], mean, **TOL, m=6))
    out.append(case("rollingIsFinite", "rollingIsFinite", [tn], core.rolling_isfinite(tn, 6).astype(float), m=6))
    q = t[10:17].copy()
    out.append(case("slidingDotProduct/short", "slidingDotProduct", [q, t], core.sliding_dot_product(q, t), **TOL))
    long = series(1500, 502)
    ql = long[100:140].copy()
    out.append(
        case("slidingDotProduct/long", "slidingDotProduct", [ql, long], core.sliding_dot_product(ql, long), **TOL)
    )
    out.append(case("zNormalize", "zNormalize", [q], core.z_norm(q), **TOL))
    out.append(case("zNormalize/constant", "zNormalize", [np.full(5, 2.0)], core.z_norm(np.full(5, 2.0)), **TOL))
    out.append(case("mass", "mass", [q, t], stumpy.mass(q, t), **TOL))
    out.append(case("mass/constant", "mass", [t[30:36].copy(), t], stumpy.mass(t[30:36].copy(), t), **TOL))
    out.append(case("mass/nan", "mass", [q, tn], stumpy.mass(q, tn), **TOL))
    out.append(case("mass/long", "mass", [ql, long], stumpy.mass(ql, long), **SELF))
    for p in [1.0, 2.0, 3.0]:
        out.append(
            case(
                f"mass/absolute/{p}",
                "mass",
                [q, t],
                stumpy.mass(q, t, normalize=False, p=p),
                **TOL,
                normalize=False,
                p=p,
            )
        )
    out.append(case("massChunked", "massChunked", [ql, long], stumpy.mass(ql, long), **SELF, chunk=300))
    for idx in [0, 20, 72]:
        d = stumpy.mass(t[idx : idx + 8].copy(), t)
        core.apply_exclusion_zone(d, idx, int(np.ceil(8 / 4)), np.inf)
        out.append(case(f"distanceProfile/{idx}", "distanceProfile", [t], d, **SELF, m=8, idx=idx))
    return out
