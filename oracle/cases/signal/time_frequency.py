"""Short-time Fourier transforms against scipy.signal (stft, istft, spectrogram) and librosa.stft; instantaneous
frequency from scipy.signal.hilbert and numpy.unwrap."""

from importlib.metadata import version
from typing import Any, Literal

import librosa
import numpy as np
import scipy
import scipy.signal as S
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import pair, tones

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "librosa": version("librosa")}

TOL = {"rtol": 1e-10, "atol": 1e-12}

type Mode = Literal["psd", "magnitude", "angle", "phase"]


def cases() -> list[Case]:
    g = rng(409)
    out: list[Case] = []
    x = tones(g, 50, fs=10.0, freqs=(1.2, 3.1))
    variants: list[dict[str, Any]] = [
        {"nperseg": 16},
        {"nperseg": 16, "noverlap": 12, "nfft": 32},
        {"nperseg": 15, "boundary": "even", "padded": False},
        {"nperseg": 16, "boundary": "odd"},
        {"nperseg": 16, "boundary": "constant", "scaling": "psd"},
        {"nperseg": 16, "boundary": None, "padded": False, "window": "hamming"},
        {"nperseg": 16, "return_onesided": False},
    ]
    for v in variants:
        label = "/".join(f"{k}={val}" for k, val in v.items())
        params: dict[str, Any] = {
            camel(k): ("none" if val is None and k == "boundary" else val) for k, val in v.items()
        }
        _, t, z = S.stft(x, fs=10.0, **v)
        out.append(case(f"stft/{label}", "stft", [x], pair(z), **TOL, fs=10.0, **params))
        out.append(case(f"stftTimes/{label}", "stftTimes", [x], t, **TOL, fs=10.0, **params))
    _, _, z = S.stft(x, nperseg=16, noverlap=8)
    _, xr = S.istft(z, nperseg=16, noverlap=8)
    out.append(case("istft", "istft", [pair(z)], xr, **TOL, nperseg=16, noverlap=8))
    _, _, z2 = S.stft(x, nperseg=12, noverlap=9, window="hann", scaling="psd", fs=10.0)
    _, xr2 = S.istft(z2, nperseg=12, noverlap=9, scaling="psd", fs=10.0)
    out.append(case("istft/psd", "istft", [pair(z2)], xr2, **TOL, nperseg=12, noverlap=9, scaling="psd", fs=10.0))
    modes: list[Mode] = ["psd", "magnitude", "angle", "phase"]
    for mode in modes:
        _, _, sx = S.spectrogram(x, fs=10.0, nperseg=16, mode=mode)
        out.append(case(f"spectrogram/{mode}", "spectrogram", [x], sx, **TOL, fs=10.0, nperseg=16, mode=mode))
    _, _, sx = S.spectrogram(x, nperseg=16, noverlap=8, window="hann", scaling="spectrum")
    out.append(
        case(
            "spectrogram/spectrum",
            "spectrogram",
            [x],
            sx,
            **TOL,
            nperseg=16,
            noverlap=8,
            window="hann",
            scaling="spectrum",
            mode="psd",
        )
    )
    d = librosa.stft(x, n_fft=16, hop_length=4)
    out.append(case("librosaStft", "librosaStft", [x], pair(d), **TOL, nFft=16, hop=4))
    d2 = librosa.stft(x, n_fft=16, hop_length=5, pad_mode="reflect")
    out.append(case("librosaStft/reflect", "librosaStft", [x], pair(d2), **TOL, nFft=16, hop=5, boundary="reflect"))
    t = np.arange(200) / 100.0
    chirp = S.chirp(t, f0=5.0, t1=2.0, f1=20.0)
    inst = np.diff(np.unwrap(np.angle(S.hilbert(chirp)))) * 100.0 / (2 * np.pi)
    out.append(case("instantaneousFrequency", "instantaneousFrequency", [chirp], inst, **TOL, fs=100.0))
    return out


def camel(name: str) -> str:
    """return_onesided -> returnOnesided."""
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)
