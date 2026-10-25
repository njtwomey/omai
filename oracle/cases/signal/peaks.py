"""Peak finding against scipy.signal (find_peaks, peak_prominences, peak_widths, argrelextrema), zero crossings
against librosa.zero_crossings and dynamic time warping against dtw-python."""

from importlib.metadata import version
from typing import Any, Literal

import librosa
import numpy as np
import scipy
import scipy.signal as S
from dtw import dtw
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "librosa": version("librosa"),
    "dtw-python": version("dtw-python"),
}

TOL = {"rtol": 1e-12, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(419)
    out: list[Case] = []
    t = np.linspace(0, 6, 120)
    x = np.sin(3 * t) + 0.6 * np.sin(7.3 * t) + 0.2 * g.normal(size=120)
    x[40:43] = x[40] + 1.0  # a plateau
    variants: list[dict[str, Any]] = [
        {},
        {"height": 0.5},
        {"height": [0.0, 1.0]},
        {"threshold": 0.05},
        {"distance": 8},
        {"prominence": 0.5},
        {"prominence": [0.3, 1.5], "wlen": 21},
        {"width": 3},
        {"width": [2, 8], "rel_height": 0.8},
        {"plateau_size": 2},
        {"height": 0.0, "distance": 5, "prominence": 0.2, "width": 1},
    ]
    for v in variants:
        peaks, props = S.find_peaks(x, **v)
        label = "/".join(f"{k}={val}" for k, val in v.items()) or "default"
        params: dict[str, Any] = {k.replace("_h", "H").replace("_s", "S"): val for k, val in v.items()}
        out.append(case(f"findPeaks/{label}", "findPeaks", [x], peaks.astype(np.float64), **TOL, **params))
        for key, arr in props.items():
            out.append(
                case(
                    f"findPeaks/{label}/{key}",
                    "findPeaksProp",
                    [x],
                    np.asarray(arr, dtype=np.float64),
                    **TOL,
                    prop=key,
                    **params,
                )
            )
    peaks, _ = S.find_peaks(x)
    prom, lb, rb = S.peak_prominences(x, peaks, wlen=15)
    out.append(
        case(
            "peakProminences",
            "peakProminences",
            [x, peaks.astype(np.float64)],
            np.stack([prom, lb, rb]),
            **TOL,
            wlen=15,
        )
    )
    w = S.peak_widths(x, peaks, rel_height=1.0)
    out.append(case("peakWidths", "peakWidths", [x, peaks.astype(np.float64)], np.stack(w), **TOL, relHeight=1.0))
    ends: list[tuple[int, Literal["clip", "wrap"]]] = [(1, "clip"), (3, "clip"), (2, "wrap")]
    for order, mode in ends:
        mx = S.argrelextrema(x, np.greater, order=order, mode=mode)[0]
        mn = S.argrelextrema(x, np.less, order=order, mode=mode)[0]
        out.append(
            case(f"argrelmax/{order}/{mode}", "argrelmax", [x], mx.astype(np.float64), **TOL, order=order, mode=mode)
        )
        out.append(
            case(f"argrelmin/{order}/{mode}", "argrelmin", [x], mn.astype(np.float64), **TOL, order=order, mode=mode)
        )
    y = np.concatenate([x[:30], np.zeros(3), x[30:40]])
    crossings: list[dict[str, Any]] = [{}, {"pad": False}, {"threshold": 0.3}, {"zero_pos": False}]
    for opts in crossings:
        z = librosa.zero_crossings(y, **opts).astype(np.float64)
        label = "/".join(f"{k}={val}" for k, val in opts.items()) or "default"
        params2: dict[str, Any] = {k.replace("_p", "P"): val for k, val in opts.items()}
        out.append(case(f"zeroCrossings/{label}", "zeroCrossings", [y], z, **TOL, **params2))
    a = np.cumsum(g.normal(size=25))
    b = np.cumsum(g.normal(size=31))
    for pattern in ["symmetric2", "symmetric1", "asymmetric"]:
        al: Any = dtw(a, b, step_pattern=pattern, keep_internals=False)
        res = np.concatenate([[al.distance, al.normalizedDistance], al.index1, al.index2]).astype(np.float64)
        out.append(case(f"dtw/{pattern}", "dtw", [a, b], res, rtol=1e-10, atol=1e-12, stepPattern=pattern))
    banded: Any = dtw(a, b, window_type="sakoechiba", window_args={"window_size": 8})
    al = banded
    res = np.concatenate([[al.distance, al.normalizedDistance], al.index1, al.index2]).astype(np.float64)
    out.append(case("dtw/window", "dtw", [a, b], res, rtol=1e-10, atol=1e-12, window=8))
    return out
