"""Spectral audio features against librosa (float64 throughout): spectrograms (centred or not, padding modes,
shorter windows), mel spectrograms, decibel conversions, MFCCs (with liftering), spectral centroid, bandwidth,
roll-off and flatness, chroma, RMS energy (from the signal and from a spectrogram), zero-crossing rate and framing."""

from importlib.metadata import version
from typing import Any

import librosa
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.signal_helpers import tones

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-10}
SR = 16000
NFFT = 512
HOP = 128


def cases() -> list[Case]:
    g = rng(450)
    out: list[Case] = []
    y = tones(g, 3000, fs=SR, freqs=(440.0, 1250.0, 3100.0), noise=0.05)
    common: dict[str, Any] = {"n_fft": NFFT, "hop_length": HOP}
    params: dict[str, Any] = {"nFft": NFFT, "hopLength": HOP}
    s2 = np.abs(librosa.stft(y, **common)) ** 2
    out.append(case("spectrogram", "spectrogram", [y], s2, **TOL, **params))
    s1 = np.abs(librosa.stft(y, **common))
    out.append(case("spectrogram/power1", "spectrogram", [y], s1, **TOL, **params, power=1))
    sc = np.abs(librosa.stft(y, center=False, **common)) ** 2
    out.append(case("spectrogram/nocenter", "spectrogram", [y], sc, **TOL, **params, center=False))
    sr_ = np.abs(librosa.stft(y, pad_mode="reflect", **common)) ** 2
    out.append(case("spectrogram/reflect", "spectrogram", [y], sr_, **TOL, **params, padMode="reflect"))
    sw = np.abs(librosa.stft(y, win_length=400, **common)) ** 2
    out.append(case("spectrogram/win", "spectrogram", [y], sw, **TOL, **params, winLength=400))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=40, dtype=np.float64, **common)
    out.append(case("melSpectrogram", "melSpectrogram", [y], mel, **TOL, **params, sr=SR, nMels=40))
    out.append(case("powerToDb", "powerToDb", [mel], librosa.power_to_db(mel), **TOL))
    out.append(
        case(
            "powerToDb/max",
            "powerToDb",
            [mel],
            librosa.power_to_db(mel, ref=np.max, top_db=None),
            **TOL,
            ref="max",
            topDb=None,
        )
    )
    out.append(case("amplitudeToDb", "amplitudeToDb", [s1], librosa.amplitude_to_db(s1, ref=np.max), **TOL, ref="max"))
    for lifter in [0, 22]:
        m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_mels=40, lifter=lifter, dtype=np.float64, **common)
        out.append(case(f"mfcc/{lifter}", "mfcc", [y], m, **TOL, **params, sr=SR, nMels=40, nMfcc=13, lifter=lifter))
    out.append(
        case("spectralCentroid", "spectralCentroid", [s1], librosa.feature.spectral_centroid(S=s1, sr=SR), **TOL, sr=SR)
    )
    for p in [2.0, 1.5]:
        bw = librosa.feature.spectral_bandwidth(S=s1, sr=SR, p=p)
        out.append(case(f"spectralBandwidth/{p}", "spectralBandwidth", [s1], bw, **TOL, sr=SR, p=p))
    for rp in [0.85, 0.5]:
        ro = librosa.feature.spectral_rolloff(S=s1, sr=SR, roll_percent=rp)
        out.append(case(f"spectralRolloff/{rp}", "spectralRolloff", [s1], ro, **TOL, sr=SR, rollPercent=rp))
    out.append(case("spectralFlatness", "spectralFlatness", [s1], librosa.feature.spectral_flatness(S=s1), **TOL))
    ch = librosa.feature.chroma_stft(y=y, sr=SR, tuning=0.0, dtype=np.float64, **common)
    out.append(case("chromaStft", "chromaStft", [y], ch, **TOL, **params, sr=SR))
    r = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP, dtype=np.float64)
    out.append(case("rms", "rms", [y], r, **TOL, frameLength=NFFT, hopLength=HOP))
    r = librosa.feature.rms(y=y, frame_length=300, hop_length=100, center=False, dtype=np.float64)
    out.append(case("rms/nocenter", "rms", [y], r, **TOL, frameLength=300, hopLength=100, center=False))
    r = librosa.feature.rms(S=s1, frame_length=NFFT, dtype=np.float64)
    out.append(case("rmsFromSpectrogram", "rmsFromSpectrogram", [s1], r, **TOL, frameLength=NFFT))
    z = librosa.feature.zero_crossing_rate(y, frame_length=256, hop_length=64)
    out.append(case("zeroCrossingRate", "zeroCrossingRate", [y], z, **TOL, frameLength=256, hopLength=64))
    out.append(
        case(
            "frame",
            "frame",
            [y[:40]],
            librosa.util.frame(y[:40], frame_length=8, hop_length=5),
            **TOL,
            frameLength=8,
            hop=5,
        )
    )
    return out
