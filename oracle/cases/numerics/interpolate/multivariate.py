"""Grid and scattered interpolation against numpy.interp and scipy.interpolate."""

from typing import Literal

import numpy as np
import scipy
import scipy.interpolate as si
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.sparse.linalg import spsolve

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

KINDS: list[Literal["linear", "nearest", "previous", "next", "zero", "slinear", "quadratic", "cubic"]] = [
    "linear",
    "nearest",
    "previous",
    "next",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
]
GRID_METHODS: list[Literal["linear", "nearest", "slinear", "cubic", "quintic"]] = [
    "linear",
    "nearest",
    "slinear",
    "cubic",
    "quintic",
]
KERNELS: list[
    Literal[
        "linear",
        "thin_plate_spline",
        "cubic",
        "quintic",
        "multiquadric",
        "inverse_multiquadric",
        "inverse_quadratic",
        "gaussian",
    ]
] = [
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]


def grid(
    axes: tuple[FloatArray, ...],
    values: FloatArray,
    method: Literal["linear", "nearest", "slinear", "cubic", "quintic"],
) -> si.RegularGridInterpolator:
    """scipy's spline methods default to an iterative sparse solver; a direct solve gives the exact interpolant."""
    if method in ("cubic", "quintic"):
        return si.RegularGridInterpolator(axes, values, method=method, solver=spsolve)
    return si.RegularGridInterpolator(axes, values, method=method)


def cases() -> list[Case]:
    g = rng(60)
    out: list[Case] = []
    xp = np.sort(g.uniform(0, 4, size=8))
    fp = g.normal(size=8)
    x = np.linspace(-1, 5, 25)
    out.append(case("interp", "interp", [x, xp, fp], np.interp(x, xp, fp), rtol=1e-12))
    out.append(
        case(
            "interp/leftright",
            "interp",
            [x, xp, fp],
            np.interp(x, xp, fp, left=-7.0, right=9.0),
            rtol=1e-12,
            left=-7.0,
            right=9.0,
        )
    )
    out.append(case("interp/period", "interp", [x, xp, fp], np.interp(x, xp, fp, period=4.5), rtol=1e-12, period=4.5))
    inside = np.linspace(float(xp[0]), float(xp[-1]), 19)
    for kind in KINDS:
        f = si.interp1d(xp, fp, kind=kind)
        out.append(case(f"interp1d/{kind}", "interp1d", [xp, fp, inside], f(inside), rtol=1e-10, atol=1e-10, kind=kind))

    ax = np.sort(g.uniform(0, 3, size=6))
    ay = np.sort(g.uniform(-1, 1, size=7))
    vals = np.sin(ax)[:, None] * np.cos(2 * ay)[None, :] + g.normal(size=(6, 7)) * 0.1
    qx = g.uniform(float(ax[0]), float(ax[-1]), size=12)
    qy = g.uniform(float(ay[0]), float(ay[-1]), size=12)
    q = np.stack([qx, qy], axis=1)
    for method in GRID_METHODS:
        f = grid((ax, ay), vals, method)
        out.append(
            case(
                f"regularGrid/2d/{method}", "regularGrid", [ax, ay, vals, q], f(q), rtol=1e-9, atol=1e-10, method=method
            )
        )
    az = np.linspace(0, 1, 5)
    vals3 = g.normal(size=(6, 7, 5))
    q3 = np.stack([qx, qy, g.uniform(0, 1, size=12)], axis=1)
    for method in ("linear", "cubic"):
        f3 = grid((ax, ay, az), vals3, method)
        out.append(
            case(
                f"regularGrid/3d/{method}",
                "regularGrid3",
                [ax, ay, az, vals3, q3],
                f3(q3),
                rtol=1e-9,
                atol=1e-10,
                method=method,
            )
        )

    sites = g.uniform(0, 1, size=(30, 2))
    sv = np.sin(3 * sites[:, 0]) + sites[:, 1] ** 2
    qs = g.uniform(-0.1, 1.1, size=(20, 2))
    for method in ("nearest", "linear"):
        out.append(
            case(
                f"griddata/{method}",
                "griddata",
                [sites, sv, qs],
                si.griddata(sites, sv, qs, method=method),
                rtol=1e-10,
                atol=1e-12,
                method=method,
            )
        )
    for kernel in KERNELS:
        epsilon = 3.0 if kernel in ("multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian") else 1.0
        r = si.RBFInterpolator(sites, sv, kernel=kernel, epsilon=epsilon)
        out.append(
            case(f"rbf/{kernel}", "rbf", [sites, sv, qs], r(qs), rtol=1e-7, atol=1e-8, kernel=kernel, epsilon=epsilon)
        )
    r = si.RBFInterpolator(sites, sv, smoothing=0.1, degree=2)
    out.append(
        case(
            "rbf/smoothed",
            "rbf",
            [sites, sv, qs],
            r(qs),
            rtol=1e-8,
            atol=1e-9,
            kernel="thin_plate_spline",
            epsilon=1.0,
            smoothing=0.1,
            degree=2,
        )
    )
    return out
