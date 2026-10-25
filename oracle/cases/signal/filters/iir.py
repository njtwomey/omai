"""IIR design against scipy.signal: prototypes, frequency transformations, the named families in ba, zpk and sos
forms, order selection, iirdesign, notch, peak and comb filters. Zeros and poles are compared as sorted lists."""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import scipy
import scipy.signal as S
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Family = Literal["butter", "cheby1", "cheby2", "ellip", "bessel"]
type Band = Literal["lowpass", "highpass", "bandpass", "bandstop"]

# scipy-stubs types zpk output per band and family; the designs here mix them.
ZPK_DESIGN = cast(Callable[..., tuple[npt.ArrayLike, npt.ArrayLike, float]], S.iirfilter)

TOL = {"rtol": 1e-9, "atol": 1e-11}


def roots(z: npt.ArrayLike) -> FloatArray:
    """Complex values sorted by real then imaginary part, as `[n, 2]`."""
    c = np.asarray(z, dtype=np.complex128).ravel()
    c = c[np.lexsort((np.round(c.imag, 9), np.round(c.real, 9)))]
    return np.stack([c.real, c.imag], axis=-1).reshape(-1, 2)


def zpk_case(name: str, fn: str, zpk: tuple[npt.ArrayLike, npt.ArrayLike, float], **params: object) -> list[Case]:
    z, p, k = zpk
    extra: dict[str, Any] = dict(params)
    return [
        case(f"{name}/z", f"{fn}Z", [], roots(z), **TOL, **extra),
        case(f"{name}/p", f"{fn}P", [], roots(p), **TOL, **extra),
        case(f"{name}/k", f"{fn}K", [], np.array(float(k)), **TOL, **extra),
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [1, 4, 5]:
        out += zpk_case(f"buttap/{n}", "buttap", S.buttap(n), n=n)
        out += zpk_case(f"cheb1ap/{n}", "cheb1ap", S.cheb1ap(n, 1.5), n=n, rp=1.5)
        out += zpk_case(f"cheb2ap/{n}", "cheb2ap", S.cheb2ap(n, 30.0), n=n, rs=30.0)
        out += zpk_case(f"ellipap/{n}", "ellipap", S.ellipap(n, 0.5, 45.0), n=n, rp=0.5, rs=45.0)
        norms: list[Literal["phase", "delay", "mag"]] = ["phase", "delay", "mag"]
        for norm in norms:
            out += zpk_case(f"besselap/{n}/{norm}", "besselap", S.besselap(n, norm=norm), n=n, norm=norm)
    out += zpk_case("ellipap/8", "ellipap", S.ellipap(8, 0.1, 80.0), n=8, rp=0.1, rs=80.0)
    out += zpk_case("besselap/12", "besselap", S.besselap(12), n=12, norm="phase")
    z, p, k = S.cheb2ap(3, 25.0)
    out += zpk_case("lp2lp", "lp2lp", S.lp2lp_zpk(z, p, k, 2.5), wo=2.5)
    out += zpk_case("lp2hp", "lp2hp", S.lp2hp_zpk(z, p, k, 2.5), wo=2.5)
    out += zpk_case("lp2bp", "lp2bp", S.lp2bp_zpk(z, p, k, 2.0, 0.7), wo=2.0, bw=0.7)
    out += zpk_case("lp2bs", "lp2bs", S.lp2bs_zpk(z, p, k, 2.0, 0.7), wo=2.0, bw=0.7)
    out += zpk_case("bilinear", "bilinear", S.bilinear_zpk(z, p, k, 3.0), fs=3.0)
    designs: list[tuple[Family, int, float | list[float], Band, dict[str, float]]] = [
        ("butter", 4, 0.3, "lowpass", {}),
        ("butter", 3, [0.2, 0.5], "bandpass", {}),
        ("cheby1", 5, 0.4, "highpass", {"rp": 1.0}),
        ("cheby2", 4, [0.2, 0.6], "bandstop", {"rs": 40.0}),
        ("ellip", 5, 0.25, "lowpass", {"rp": 0.5, "rs": 50.0}),
        ("ellip", 4, [0.3, 0.45], "bandpass", {"rp": 1.0, "rs": 60.0}),
        ("bessel", 4, 0.2, "lowpass", {}),
    ]
    for ftype, n, wn, btype, ripple in designs:
        label = f"{ftype}/{n}/{btype}"
        rp, rs = ripple.get("rp"), ripple.get("rs")
        params: dict[str, Any] = {"n": n, "wn": wn, "btype": btype, "ftype": ftype, **ripple}
        b, a = S.iirfilter(n, wn, rp=rp, rs=rs, btype=btype, ftype=ftype, output="ba")
        out.append(case(f"{label}/b", "iirB", [], b, **TOL, **params))
        out.append(case(f"{label}/a", "iirA", [], a, **TOL, **params))
        sos = S.iirfilter(n, wn, rp=rp, rs=rs, btype=btype, ftype=ftype, output="sos")
        out.append(case(f"{label}/sos", "iirSos", [], sos, **TOL, **params))
        out += zpk_case(
            f"{label}/zpk", "iirZpk", ZPK_DESIGN(n, wn, rp=rp, rs=rs, btype=btype, ftype=ftype, output="zpk"), **params
        )
    b, a = S.butter(3, 1000.0, fs=8000.0)
    out.append(case("butter/fs/b", "iirB", [], b, **TOL, n=3, wn=1000.0, fs=8000.0, ftype="butter", btype="lowpass"))
    b, a = S.cheby1(4, 2.0, [3.0, 7.0], btype="bandpass", analog=True)
    params2: dict[str, Any] = {
        "n": 4,
        "wn": [3.0, 7.0],
        "btype": "bandpass",
        "ftype": "cheby1",
        "rp": 2.0,
        "analog": True,
    }
    out.append(case("cheby1/analog/b", "iirB", [], b, **TOL, **params2))
    out.append(case("cheby1/analog/a", "iirA", [], a, **TOL, **params2))
    specs: list[tuple[float | list[float], float | list[float], bool]] = [
        (0.2, 0.3, False),
        (0.3, 0.2, False),
        ([0.2, 0.5], [0.1, 0.6], False),
        ([0.1, 0.6], [0.2, 0.5], False),
        (10.0, 30.0, True),
    ]
    for wp, ws, analog in specs:
        for name, f in [
            ("buttord", S.buttord),
            ("cheb1ord", S.cheb1ord),
            ("cheb2ord", S.cheb2ord),
            ("ellipord", S.ellipord),
        ]:
            n, wn = f(wp, ws, 1.0, 45.0, analog=analog)
            res = np.concatenate([[float(n)], np.atleast_1d(np.asarray(wn, dtype=np.float64))])
            # Bandstop orders refine the passband edges by fminbound to xatol 1e-5, so the edges agree to ~1e-7.
            rtol = 1e-6 if isinstance(wp, list) and isinstance(ws, list) and wp[0] < ws[0] else 1e-9
            out.append(
                case(f"{name}/{wp}/{ws}/{analog}", name, [], res, rtol=rtol, atol=1e-9, wp=wp, ws=ws, analog=analog)
            )
    n, wn = S.ellipord(1000.0, 1500.0, 0.5, 60.0, fs=8000.0)
    res = np.array([float(n), float(wn)])
    out.append(
        case(
            "ellipord/fs",
            "ellipord",
            [],
            res,
            rtol=1e-7,
            atol=1e-9,
            wp=1000.0,
            ws=1500.0,
            gpass=0.5,
            gstop=60.0,
            fs=8000.0,
        )
    )
    sosd = S.iirdesign([0.2, 0.5], [0.1, 0.6], 1.0, 40.0, ftype="cheby2", output="sos")
    out.append(case("iirdesign", "iirdesign", [], sosd, **TOL, wp=[0.2, 0.5], ws=[0.1, 0.6], ftype="cheby2"))
    for name, f in [("iirnotch", S.iirnotch), ("iirpeak", S.iirpeak)]:
        b, a = f(60.0, 30.0, fs=1000.0)
        out.append(case(f"{name}/b", f"{name}B", [], b, **TOL, w0=60.0, q=30.0, fs=1000.0))
        out.append(case(f"{name}/a", f"{name}A", [], a, **TOL, w0=60.0, q=30.0, fs=1000.0))
    kinds: list[Literal["notch", "peak"]] = ["notch", "peak"]
    for ftype in kinds:
        for pass_zero in [False, True]:
            b, a = S.iircomb(100.0, 20.0, ftype=ftype, fs=1000.0, pass_zero=pass_zero)
            res = np.stack([b, a])
            out.append(case(f"iircomb/{ftype}/{pass_zero}", "iircomb", [], res, **TOL, ftype=ftype, passZero=pass_zero))
    return out
