"""Pitch estimation against librosa: autocorrelation, YIN and probabilistic YIN (f0 with NaN for unvoiced frames,
voicing flags and probabilities) on a gliding tone with a silent gap."""

from importlib.metadata import version

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "numpy": np.__version__}

SR = 8000


def glide(g: np.random.Generator) -> FloatArray:
    n = 6000
    t = np.arange(n) / SR
    f = 180 + 120 * t / t[-1]
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.01 * g.normal(size=n)
    y[2500:3300] = 0.002 * g.normal(size=800)
    return np.asarray(y, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(460)
    out: list[Case] = []
    x = g.normal(size=50)
    out.append(
        case(
            "autocorrelate",
            "autocorrelate",
            [x],
            librosa.autocorrelate(x, max_size=20),
            rtol=1e-10,
            atol=1e-12,
            maxSize=20,
        )
    )
    y = glide(g)
    for thr in [0.1, 0.25]:
        f0 = librosa.yin(y, fmin=80, fmax=600, sr=SR, frame_length=512, hop_length=128, trough_threshold=thr)
        out.append(
            case(
                f"yin/{thr}",
                "yin",
                [y],
                f0,
                rtol=1e-9,
                atol=1e-9,
                fmin=80,
                fmax=600,
                sr=SR,
                frameLength=512,
                hopLength=128,
                troughThreshold=thr,
            )
        )
    f0, voiced, prob = librosa.pyin(y, fmin=80, fmax=600, sr=SR, frame_length=512, hop_length=128)
    res = np.concatenate([f0, voiced.astype(float), prob])
    out.append(
        case("pyin", "pyin", [y], res, rtol=1e-9, atol=1e-9, fmin=80, fmax=600, sr=SR, frameLength=512, hopLength=128)
    )
    return out
