"""Perceptual frequency scales: mel conversions and frequencies against librosa; Bark (Traunmüller,
Zwicker-Terhardt, Schroeder) and ERB (Glasberg-Moore) from their published formulas in numpy (no reference library
implements all of them)."""

from importlib.metadata import version

import librosa
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"librosa": version("librosa"), "numpy": np.__version__}

TOL = {"rtol": 1e-12, "atol": 1e-12}


def traunmuller(f: FloatArray) -> FloatArray:
    z = 26.81 * f / (1960 + f) - 0.53
    z = np.where(z < 2, z + 0.15 * (2 - z), z)
    return np.where(z >= 20.1, z + 0.22 * (z - 20.1), z)


def cases() -> list[Case]:
    out: list[Case] = []
    f = np.array([0.0, 50.0, 300.0, 999.0, 1000.0, 1001.0, 4000.0, 11025.0, 20000.0])
    for htk in [False, True]:
        out.append(case(f"hzToMel/{htk}", "hzToMel", [f], librosa.hz_to_mel(f, htk=htk), **TOL, htk=htk))
        m = np.linspace(0, 60, 13)
        out.append(case(f"melToHz/{htk}", "melToHz", [m], librosa.mel_to_hz(m, htk=htk), **TOL, htk=htk))
        mf = librosa.mel_frequencies(n_mels=10, fmin=30.0, fmax=8000.0, htk=htk)
        out.append(
            case(
                f"melFrequencies/{htk}",
                "melFrequencies",
                [],
                mf,
                rtol=1e-11,
                atol=1e-10,
                n=10,
                fmin=30.0,
                fmax=8000.0,
                htk=htk,
            )
        )
    out.append(
        case(
            "fftFrequencies",
            "fftFrequencies",
            [],
            librosa.fft_frequencies(sr=16000, n_fft=64),
            **TOL,
            sr=16000,
            nfft=64,
        )
    )
    out.append(case("hzToBark/traunmuller", "hzToBark", [f], traunmuller(f), **TOL, formula="traunmuller"))
    zw = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
    out.append(case("hzToBark/zwicker", "hzToBark", [f], zw, **TOL, formula="zwicker"))
    out.append(case("hzToBark/schroeder", "hzToBark", [f], 7 * np.arcsinh(f / 650), **TOL, formula="schroeder"))
    out.append(case("erbBandwidth", "erbBandwidth", [f], 24.7 * (4.37e-3 * f + 1), **TOL))
    out.append(case("hzToErb", "hzToErb", [f], 21.4 * np.log10(1 + 4.37e-3 * f), **TOL))
    return out
