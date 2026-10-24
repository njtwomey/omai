"""Augmentation against librosa: the phase vocoder, time stretching and pitch shifting (FFT resampling); mixing at
an SNR from its definition."""

from importlib.metadata import version

import librosa
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.signal_helpers import pair, tones

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-10}


def cases() -> list[Case]:
    g = rng(530)
    out: list[Case] = []
    y = tones(g, 3000, fs=8000.0, freqs=(330.0, 870.0), noise=0.02)
    d = librosa.stft(y, n_fft=256, hop_length=64)
    for rate in [1.5, 0.7]:
        pv = librosa.phase_vocoder(d, rate=rate)
        out.append(
            case(f"phaseVocoder/{rate}", "phaseVocoder", [y], pair(pv), **TOL, rate=rate, nFft=256, hopLength=64)
        )
        ts = librosa.effects.time_stretch(y, rate=rate, n_fft=256, hop_length=64)
        out.append(case(f"timeStretch/{rate}", "timeStretch", [y], ts, **TOL, rate=rate, nFft=256, hopLength=64))
    ps = librosa.effects.pitch_shift(y, sr=8000, n_steps=3, res_type="fft", n_fft=256, hop_length=64)
    out.append(
        case("pitchShift", "pitchShift", [y], ps, rtol=1e-8, atol=1e-9, sr=8000, nSteps=3, nFft=256, hopLength=64)
    )
    noise = g.normal(size=3000)
    ps_ = np.sum(y**2)
    pn = np.sum(noise**2)
    mixed = y + np.sqrt(ps_ / (pn * 10 ** (5 / 10))) * noise
    out.append(case("mixAtSnr", "mixAtSnr", [y, noise], mixed, rtol=1e-12, atol=1e-12, snrDb=5))
    return out
