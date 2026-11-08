"""The constant-Q transform against librosa.cqt over one octave (where librosa resamples nothing) and its filters
against librosa.filters.wavelet."""

from importlib.metadata import version
from typing import Any, Literal

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.signal_helpers import pair

REFERENCE: dict[str, str] = {"librosa": version("librosa")}

TOL = {"rtol": 1e-9, "atol": 1e-11}
# librosa builds the CQT basis in complex64 (filters.wavelet's default dtype), so the transform agrees to ~1e-7.
CQT_TOL = {"rtol": 1e-6, "atol": 1e-7}


def cases() -> list[Case]:
    g = rng(411)
    out: list[Case] = []
    sr = 8000.0
    t = np.arange(1500) / sr
    y = np.cos(2 * np.pi * 2500 * t) + 0.5 * np.sin(2 * np.pi * 3100 * t) + 0.05 * g.normal(size=1500)
    pads: list[tuple[int, Literal["constant", "reflect"]]] = [(128, "constant"), (200, "reflect")]
    for hop, pad in pads:
        c = librosa.cqt(
            y, sr=sr, hop_length=hop, fmin=2000.0, n_bins=12, bins_per_octave=12, sparsity=0.0, tuning=0.0, pad_mode=pad
        )
        params: dict[str, Any] = {"sr": sr, "hop": hop, "fmin": 2000.0, "nBins": 12, "padMode": pad}
        out.append(case(f"cqt/{hop}/{pad}", "cqt", [y], pair(c), **CQT_TOL, **params))
    c = librosa.cqt(
        y, sr=sr, hop_length=64, fmin=2400.0, n_bins=6, bins_per_octave=24, sparsity=0.0, tuning=0.0, filter_scale=0.8
    )
    out.append(
        case(
            "cqt/24",
            "cqt",
            [y],
            pair(c),
            **CQT_TOL,
            sr=sr,
            hop=64,
            fmin=2400.0,
            nBins=6,
            binsPerOctave=24,
            filterScale=0.8,
        )
    )
    freqs = librosa.cqt_frequencies(n_bins=12, fmin=2000.0)
    basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=sr, dtype=np.complex128)
    out.append(case("filters", "cqtFilters", [], pair(basis), **TOL, sr=sr, fmin=2000.0, nBins=12))
    out.append(case("lengths", "cqtLengths", [], lengths, **TOL, sr=sr, fmin=2000.0, nBins=12))
    return out
