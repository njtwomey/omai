"""Auditory filter banks: mel and chroma filters against librosa.filters (float64), gammatone filters against
scipy.signal.gammatone, and a gammatone bank applied by scipy.signal.lfilter."""

from importlib.metadata import version
from typing import Any, Literal

import librosa
import numpy as np
import scipy
import scipy.signal as S
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "scipy": scipy.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(440)
    out: list[Case] = []
    variants: list[dict[str, Any]] = [
        {"sr": 16000, "n_fft": 512, "n_mels": 40},
        {"sr": 22050, "n_fft": 256, "n_mels": 20, "fmin": 100.0, "fmax": 6000.0, "htk": True},
        {"sr": 8000, "n_fft": 128, "n_mels": 16, "norm": None},
        {"sr": 8000, "n_fft": 128, "n_mels": 16, "norm": 2.0},
    ]
    for i, v in enumerate(variants):
        w = librosa.filters.mel(dtype=np.float64, **v)
        params: dict[str, Any] = {k: v[k] for k in v if k not in ("sr", "n_fft")}
        out.append(case(f"melFilterBank/{i}", "melFilterBank", [], w, **TOL, sr=v["sr"], nfft=v["n_fft"], **params))
    chroma_variants: list[dict[str, Any]] = [
        {"sr": 22050, "n_fft": 512},
        {"sr": 16000, "n_fft": 256, "n_chroma": 24, "tuning": 0.2, "octwidth": None, "norm": 1.0, "base_c": False},
    ]
    for i, v in enumerate(chroma_variants):
        w = librosa.filters.chroma(dtype=np.float64, **v)
        params = {k: v[k] for k in v if k not in ("sr", "n_fft")}
        out.append(
            case(f"chromaFilterBank/{i}", "chromaFilterBank", [], w, **TOL, sr=v["sr"], nfft=v["n_fft"], **params)
        )
    ftypes: list[Literal["fir", "iir"]] = ["fir", "iir"]
    for ftype in ftypes:
        for freq, fs in [(440.0, 16000.0), (2500.0, 22050.0)]:
            b, a = S.gammatone(freq, ftype, fs=fs)
            out.append(
                case(
                    f"gammatone/{ftype}/{freq}",
                    "gammatone",
                    [],
                    np.concatenate([b, a]),
                    **TOL,
                    freq=freq,
                    ftype=ftype,
                    fs=fs,
                )
            )
    b, a = S.gammatone(1000.0, "fir", order=6, numtaps=40, fs=8000.0)
    out.append(
        case(
            "gammatone/fir/order",
            "gammatone",
            [],
            np.concatenate([b, a]),
            **TOL,
            freq=1000.0,
            ftype="fir",
            fs=8000.0,
            order=6,
            numtaps=40,
        )
    )
    x = g.normal(size=200)
    centres = [300.0, 900.0, 2000.0]
    bands = [S.lfilter(*S.gammatone(f, "iir", fs=8000.0), x) for f in centres]
    # The IIR gammatone has a fourfold repeated pole pair near the unit circle, so its response is hypersensitive to
    # the last bits of the coefficients (repeated roots move as eps^(1/4)): outputs agree to 1e-8 relative, 1e-6
    # near zero crossings.
    out.append(
        case("gammatoneBank", "gammatoneBank", [x], np.stack(bands), rtol=1e-6, atol=1e-8, centres=centres, fs=8000.0)
    )
    return out
