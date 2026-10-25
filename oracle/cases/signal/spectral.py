"""Spectral estimation against scipy.signal (detrend, periodogram, welch, csd, coherence) and chi-square intervals
against scipy.stats.chi2."""

from typing import Any

import numpy as np
import scipy
import scipy.signal as S
import scipy.stats as st
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import pair, tones

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-13}


def cases() -> list[Case]:
    g = rng(406)
    out: list[Case] = []
    x = tones(g, 64, fs=8.0, freqs=(0.9, 2.2)) + 0.05 * np.arange(64)
    y = 0.5 * x + 0.3 * g.normal(size=64)
    m = g.normal(size=(3, 40))
    out.append(case("detrend/linear", "detrend", [x], S.detrend(x), **TOL))
    out.append(case("detrend/constant", "detrend", [x], S.detrend(x, type="constant"), **TOL, type="constant"))
    out.append(case("detrend/bp", "detrend", [x], S.detrend(x, bp=[20, 45]), **TOL, bp=[20, 45]))
    out.append(case("detrend/axis0", "detrend", [m.T], S.detrend(m.T, axis=0), **TOL, axis=0))
    variants: list[dict[str, Any]] = [
        {},
        {"window": "hann"},
        {"nfft": 100},
        {"nfft": 50},
        {"detrend": "linear", "scaling": "spectrum"},
        {"return_onesided": False},
    ]
    for v in variants:
        f, p = S.periodogram(x, fs=8.0, **v)
        label = "/".join(f"{k}={val}" for k, val in v.items()) or "default"
        params: dict[str, Any] = {snake_to_camel(k): val for k, val in v.items()}
        out.append(case(f"periodogram/{label}", "periodogram", [x], p, **TOL, fs=8.0, **params))
        out.append(case(f"periodogramFreqs/{label}", "periodogramFreqs", [x], f, **TOL, fs=8.0, **params))
    wvariants: list[dict[str, Any]] = [
        {"nperseg": 16},
        {"nperseg": 16, "noverlap": 4, "nfft": 32},
        {"nperseg": 20, "window": ("tukey", 0.3), "average": "median"},
        {"nperseg": 16, "detrend": "linear", "scaling": "spectrum"},
        {"nperseg": 15, "return_onesided": False},
        {"nperseg": 100},
    ]
    for v in wvariants:
        label = "/".join(f"{k}={val}" for k, val in v.items())
        params: dict[str, Any] = {
            snake_to_camel(k): (list(val) if isinstance(val, tuple) else val) for k, val in v.items()
        }
        _, p = S.welch(x, fs=8.0, **v)
        out.append(case(f"welch/{label}", "welch", [x], p, **TOL, fs=8.0, **params))
        _, c = S.csd(x, y, fs=8.0, **v)
        out.append(case(f"csd/{label}", "csd", [x, y], pair(c), **TOL, fs=8.0, **params))
    _, p = S.welch(m, nperseg=10, axis=-1)
    out.append(case("welch/batch", "welch", [m], p, **TOL, nperseg=10))
    _, p = S.welch(m.T, nperseg=10, axis=0)
    out.append(case("welch/axis0", "welch", [m.T], p, **TOL, nperseg=10, axis=0))
    _, c = S.csd(x, y[:50], nperseg=16)
    out.append(case("csd/unequal", "csd", [x, y[:50]], pair(c), **TOL, nperseg=16))
    _, coh = S.coherence(x, y, fs=8.0, nperseg=16)
    out.append(case("coherence", "coherence", [x, y], coh, **TOL, fs=8.0, nperseg=16))
    psd = np.abs(g.normal(size=5)) + 0.1
    for dof, level in [(2.0, 0.95), (10.0, 0.9), (37.5, 0.99)]:
        lo = dof * psd / st.chi2.ppf(1 - (1 - level) / 2, dof)
        hi = dof * psd / st.chi2.ppf((1 - level) / 2, dof)
        out.append(
            case(
                f"confidence/{dof}/{level}",
                "confidence",
                [psd],
                np.stack([lo, hi]),
                rtol=1e-9,
                atol=1e-12,
                dof=dof,
                level=level,
            )
        )
    return out


def snake_to_camel(name: str) -> str:
    """return_onesided -> returnOnesided."""
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)
