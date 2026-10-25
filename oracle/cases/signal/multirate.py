"""Multirate processing against scipy.signal: upfirdn (every extension mode), resample_poly, resample and decimate."""

from collections.abc import Callable
from typing import Any, Literal, cast

import scipy
import scipy.signal as S
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.signal_helpers import tones

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

# scipy-stubs omits some of upfirdn's extension modes.
UPFIRDN = cast(Callable[..., FloatArray], S.upfirdn)

TOL = {"rtol": 1e-10, "atol": 1e-12}

type Mode = Literal[
    "constant", "symmetric", "reflect", "edge", "wrap", "antisymmetric", "antireflect", "smooth", "line"
]
MODES: list[Mode] = [
    "constant",
    "symmetric",
    "reflect",
    "edge",
    "wrap",
    "antisymmetric",
    "antireflect",
    "smooth",
    "line",
]


def cases() -> list[Case]:
    g = rng(405)
    out: list[Case] = []
    x = tones(g, 23)
    h = g.normal(size=7)
    for up, down in [(1, 1), (3, 1), (1, 2), (3, 2), (2, 5)]:
        y = S.upfirdn(h, x, up, down)
        out.append(case(f"upfirdn/{up}/{down}", "upfirdn", [h, x], y, **TOL, up=up, down=down))
    for mode in MODES:
        params: dict[str, Any] = {"up": 2, "down": 3, "mode": mode, "cval": 0.7}
        y = UPFIRDN(h, x, 2, 3, mode=mode, cval=0.7)
        out.append(case(f"upfirdn/mode/{mode}", "upfirdn", [h, x], y, **TOL, **params))
    m = g.normal(size=(9, 3))
    out.append(case("upfirdn/axis0", "upfirdn", [h, m], S.upfirdn(h, m, 2, 1, axis=0), **TOL, up=2, down=1, axis=0))
    pads: list[Literal["constant", "mean", "median", "minimum", "maximum", "line", "symmetric"]] = [
        "constant",
        "mean",
        "median",
        "minimum",
        "maximum",
        "line",
        "symmetric",
    ]
    for up, down in [(3, 2), (1, 4), (5, 3)]:
        for padtype in pads:
            y = S.resample_poly(x + 2.0, up, down, padtype=padtype)
            params2: dict[str, Any] = {"up": up, "down": down, "padtype": padtype}
            out.append(case(f"resamplePoly/{up}/{down}/{padtype}", "resamplePoly", [x + 2.0], y, **TOL, **params2))
    y = S.resample_poly(x, 3, 2, window=("kaiser", 8.0), padtype="constant", cval=1.5)
    out.append(
        case("resamplePoly/window", "resamplePoly", [x], y, **TOL, up=3, down=2, window=["kaiser", 8.0], cval=1.5)
    )
    taps = S.firwin(15, 0.3)
    y = S.resample_poly(x, 2, 3, window=taps)
    out.append(case("resamplePoly/taps", "resamplePolyTaps", [x, taps], y, **TOL, up=2, down=3))
    for num in [11, 40, 23, 46]:
        out.append(case(f"resample/{num}", "resample", [x], S.resample(x, num), **TOL, num=num))
    out.append(case("resample/window", "resample", [x], S.resample(x, 30, window="hann"), **TOL, num=30, window="hann"))
    long = tones(g, 80)
    ftypes: list[Literal["iir", "fir"]] = ["iir", "fir"]
    for ftype in ftypes:
        for zero_phase in [True, False]:
            y = S.decimate(long, 4, ftype=ftype, zero_phase=zero_phase)
            params3: dict[str, Any] = {"q": 4, "ftype": ftype, "zeroPhase": zero_phase}
            out.append(case(f"decimate/{ftype}/{zero_phase}", "decimate", [long], y, rtol=1e-9, atol=1e-11, **params3))
    return out
