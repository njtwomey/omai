"""Rhythm against librosa (float64 mel filters): onset strength (mean and median aggregation, lags), peak picking
and onset detection, the tempogram, tempo estimation and dynamic-programming beat tracking, on a click track with
noise."""

from importlib.metadata import version

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "numpy": np.__version__}

SR = 22050
TOL = {"rtol": 1e-9, "atol": 1e-9}


def clicks(g: np.random.Generator, bpm: float, seconds: float) -> FloatArray:
    n = int(SR * seconds)
    y = 0.01 * g.normal(size=n)
    period = int(SR * 60 / bpm)
    t = np.arange(400)
    burst = np.sin(2 * np.pi * 1500 * t / SR) * np.exp(-t / 80)
    for s in range(period // 3, n - 400, period):
        y[s : s + 400] += burst
    return y


def cases() -> list[Case]:
    g = rng(520)
    out: list[Case] = []
    y = clicks(g, 125.0, 4.0)
    env = librosa.onset.onset_strength(y=y, sr=SR, dtype=np.float64)
    out.append(case("onsetStrength", "onsetStrength", [y], env, **TOL))
    med = librosa.onset.onset_strength(y=y, sr=SR, aggregate=np.median, lag=2, dtype=np.float64)
    out.append(case("onsetStrength/median", "onsetStrength", [y], med, **TOL, aggregate="median", lag=2))
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.5, wait=4)
    out.append(
        case(
            "peakPick", "peakPick", [env], pk.astype(float), preMax=3, postMax=3, preAvg=5, postAvg=5, delta=0.5, wait=4
        )
    )
    on = librosa.onset.onset_detect(onset_envelope=env, sr=SR)
    out.append(case("onsetDetect", "onsetDetect", [env], on.astype(float), sr=SR, hopLength=512))
    tg = librosa.feature.tempogram(onset_envelope=env, sr=SR, win_length=64)
    out.append(case("tempogram", "tempogram", [env], tg, **TOL, winLength=64))
    bpm = librosa.feature.tempo(onset_envelope=env, sr=SR)
    out.append(case("tempo", "tempo", [env], np.atleast_1d(bpm), **TOL))
    medbeat = librosa.onset.onset_strength(y=y, sr=SR, aggregate=np.median, dtype=np.float64)
    tempo_, beats = librosa.beat.beat_track(onset_envelope=medbeat, sr=SR)
    out.append(
        case("beatTrack", "beatTrack", [medbeat], np.concatenate([np.atleast_1d(tempo_), beats.astype(float)]), **TOL)
    )
    tempo_, beats = librosa.beat.beat_track(onset_envelope=medbeat, sr=SR, tightness=400, trim=False, bpm=100.0)
    out.append(
        case(
            "beatTrack/fixed",
            "beatTrack",
            [medbeat],
            np.concatenate([np.atleast_1d(tempo_), beats.astype(float)]),
            **TOL,
            tightness=400,
            trim=False,
            bpm=100.0,
        )
    )
    return out
