"""Reassigned spectrograms against librosa.reassigned_spectrogram."""

from importlib.metadata import version
from typing import Any, Literal

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"librosa": version("librosa")}

TOL = {"rtol": 1e-8, "atol": 1e-9}


def cases() -> list[Case]:
    g = rng(410)
    out: list[Case] = []
    sr = 8000.0
    t = np.arange(600) / sr
    y = np.cos(2 * np.pi * (600 * t + 900 * t**2)) + 0.5 * np.cos(2 * np.pi * 2200 * t) + 0.01 * g.normal(size=600)
    setups: list[tuple[bool, Literal["constant", "reflect"]]] = [
        (True, "constant"),
        (True, "reflect"),
        (False, "constant"),
    ]
    for center, pad in setups:
        f, tt, m = librosa.reassigned_spectrogram(y, sr=sr, n_fft=64, hop_length=16, center=center, pad_mode=pad)
        params: dict[str, Any] = {"sr": sr, "nFft": 64, "hop": 16, "center": center, "padMode": pad}
        label = f"{center}/{pad}"
        out.append(case(f"frequencies/{label}", "reassignedFrequencies", [y], f, **TOL, **params))
        out.append(case(f"times/{label}", "reassignedTimes", [y], tt, **TOL, **params))
        out.append(case(f"magnitudes/{label}", "reassignedMagnitudes", [y], m, **TOL, **params))
    f, tt, _ = librosa.reassigned_spectrogram(y, sr=sr, n_fft=64, hop_length=16, fill_nan=True, ref_power=1e-2)
    params2 = {"sr": sr, "nFft": 64, "hop": 16, "fillNan": True, "refPower": 1e-2}
    out.append(case("frequencies/fill", "reassignedFrequencies", [y], f, **TOL, **params2))
    out.append(case("times/fill", "reassignedTimes", [y], tt, **TOL, **params2))
    return out
