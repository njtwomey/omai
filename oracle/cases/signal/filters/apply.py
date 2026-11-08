"""Filter application against scipy.signal (lfilter, sosfilt, filtfilt, initial conditions, medfilt, Savitzky-Golay),
numpy.convolve moving averages and pandas exponentially weighted means."""

from typing import Literal

import numpy as np
import pandas as pd
import scipy
import scipy.signal as S
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.signal_helpers import tones

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "pandas": pd.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(403)
    out: list[Case] = []
    x = tones(g, 40)
    b, a = S.butter(4, 0.2)
    out.append(case("lfilter/iir", "lfilter", [b, a, x], S.lfilter(b, a, x), **TOL))
    out.append(
        case(
            "lfilter/fir",
            "lfilter",
            [np.array([0.2, 0.5, 0.3]), np.array([1.0]), x],
            S.lfilter([0.2, 0.5, 0.3], [1.0], x),
            **TOL,
        )
    )
    a2 = np.array([2.0, -0.6, 0.1])
    b2 = np.array([1.0, 0.4])
    out.append(case("lfilter/unnormalised", "lfilter", [b2, a2, x], S.lfilter(b2, a2, x), **TOL))
    zi = g.normal(size=4)
    y, zf = S.lfilter(b, a, x, zi=zi)
    out.append(case("lfilter/zi/y", "lfilterY", [b, a, x, zi], y, **TOL))
    out.append(case("lfilter/zi/zf", "lfilterZf", [b, a, x, zi], zf, **TOL))
    short = x[:2]
    y, zf = S.lfilter(b, a, short, zi=zi)
    out.append(case("lfilter/short/y", "lfilterY", [b, a, short, zi], y, **TOL))
    out.append(case("lfilter/short/zf", "lfilterZf", [b, a, short, zi], zf, **TOL))
    m = g.normal(size=(12, 3))
    out.append(case("lfilter/axis0", "lfilter", [b, a, m], S.lfilter(b, a, m, axis=0), **TOL, axis=0))
    out.append(case("lfilterZi", "lfilterZi", [b, a], S.lfilter_zi(b, a), **TOL))
    out.append(case("lfilterZi/unnormalised", "lfilterZi", [b2, a2], S.lfilter_zi(b2, a2), **TOL))
    yp = np.array([0.5, -0.2, 0.1])
    xp = np.array([1.0, 0.3])
    out.append(case("lfiltic", "lfiltic", [b, a, yp, xp], S.lfiltic(b, a, yp, xp), **TOL))
    out.append(case("lfiltic/unnormalised", "lfiltic", [b2, a2, yp], S.lfiltic(b2, a2, yp), **TOL))
    sos = S.butter(6, [0.1, 0.4], btype="band", output="sos")
    out.append(case("sosfilt", "sosfilt", [sos, x], S.sosfilt(sos, x), **TOL))
    szi = g.normal(size=(sos.shape[0], 2))
    y, zf = S.sosfilt(sos, x, zi=szi)
    out.append(case("sosfilt/zi/y", "sosfiltY", [sos, x, szi], y, **TOL))
    out.append(case("sosfilt/zi/zf", "sosfiltZf", [sos, x, szi], zf, **TOL))
    out.append(case("sosfiltZi", "sosfiltZi", [sos], S.sosfilt_zi(sos), **TOL))
    pads: list[Literal["odd", "even", "constant"] | None] = ["odd", "even", "constant", None]
    for padtype in pads:
        name = padtype or "none"
        out.append(
            case(f"filtfilt/{name}", "filtfilt", [b, a, x], S.filtfilt(b, a, x, padtype=padtype), **TOL, padtype=name)
        )
    out.append(
        case("filtfilt/padlen", "filtfilt", [b, a, x], S.filtfilt(b, a, x, padlen=7), **TOL, padtype="odd", padlen=7)
    )
    out.append(
        case(
            "filtfilt/gust",
            "filtfilt",
            [b, a, x],
            S.filtfilt(b, a, x, method="gust"),
            rtol=1e-8,
            atol=1e-10,
            method="gust",
        )
    )
    long = tones(g, 60)
    gi = S.filtfilt(b, a, long, method="gust", irlen=20)
    out.append(
        case("filtfilt/gust/irlen", "filtfilt", [b, a, long], gi, rtol=1e-8, atol=1e-10, method="gust", irlen=20)
    )
    m2 = g.normal(size=(30, 2))
    out.append(case("filtfilt/axis0", "filtfilt", [b, a, m2], S.filtfilt(b, a, m2, axis=0), **TOL, axis=0))
    out.append(case("sosfiltfilt", "sosfiltfilt", [sos, x], S.sosfiltfilt(sos, x), **TOL))
    out.append(
        case(
            "sosfiltfilt/even",
            "sosfiltfilt",
            [sos, x],
            S.sosfiltfilt(sos, x, padtype="even", padlen=10),
            **TOL,
            padtype="even",
            padlen=10,
        )
    )
    spiky = x.copy()
    spiky[[5, 17, 30]] += 6
    out.append(case("medfilt/1d", "medfilt", [spiky], S.medfilt(spiky, 5), kernel=5))
    img = g.normal(size=(6, 7))
    out.append(case("medfilt/2d", "medfilt", [img], S.medfilt(img, [3, 5]), kernel=[3, 5]))
    for wl, po, deriv in [(5, 2, 0), (7, 3, 0), (7, 3, 1), (6, 2, 0), (9, 4, 2)]:
        c = S.savgol_coeffs(wl, po, deriv=deriv, delta=0.5)
        out.append(
            case(
                f"savgolCoeffs/{wl}/{po}/{deriv}",
                "savgolCoeffs",
                [],
                c,
                **TOL,
                window=wl,
                order=po,
                deriv=deriv,
                delta=0.5,
            )
        )
    out.append(
        case(
            "savgolCoeffs/dot/pos",
            "savgolCoeffs",
            [],
            S.savgol_coeffs(7, 2, pos=2, use="dot"),
            **TOL,
            window=7,
            order=2,
            pos=2,
            use="dot",
        )
    )
    modes: list[Literal["interp", "mirror", "constant", "nearest", "wrap"]] = [
        "interp",
        "mirror",
        "constant",
        "nearest",
        "wrap",
    ]
    for mode in modes:
        for wl, po, deriv in [(7, 2, 0), (8, 3, 1)]:
            y = S.savgol_filter(x, wl, po, deriv=deriv, mode=mode, cval=0.5)
            out.append(
                case(
                    f"savgolFilter/{mode}/{wl}/{deriv}",
                    "savgolFilter",
                    [x],
                    y,
                    **TOL,
                    window=wl,
                    order=po,
                    deriv=deriv,
                    mode=mode,
                    cval=0.5,
                )
            )
    out.append(
        case(
            "savgolFilter/axis0",
            "savgolFilter",
            [m],
            S.savgol_filter(m, 5, 2, axis=0),
            **TOL,
            window=5,
            order=2,
            deriv=0,
            mode="interp",
            axis=0,
        )
    )
    conv: list[Literal["valid", "same", "full"]] = ["valid", "same", "full"]
    for mode in conv:
        out.append(
            case(
                f"movingAverage/{mode}",
                "movingAverage",
                [x],
                np.convolve(x, np.ones(4) / 4, mode=mode),
                **TOL,
                window=4,
                mode=mode,
            )
        )
    for adjust in [True, False]:
        ew = pd.Series(x).ewm(alpha=0.3, adjust=adjust).mean().to_numpy()
        out.append(case(f"ewm/{adjust}", "exponentialMovingAverage", [x], ew, **TOL, alpha=0.3, adjust=adjust))
    return out
