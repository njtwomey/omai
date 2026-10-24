"""Lomb-Scargle periodograms against scipy.signal.lombscargle and false-alarm probabilities against astropy."""

from importlib.metadata import version
from typing import Literal

import numpy as np
import scipy
import scipy.signal as S
from astropy.timeseries.periodograms.lombscargle import _statistics as A
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.signal_helpers import pair

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "astropy": version("astropy")}

TOL = {"rtol": 1e-10, "atol": 1e-12}

type Method = Literal["single", "naive", "davies", "baluev"]
METHODS: list[Method] = ["single", "naive", "davies", "baluev"]


def cases() -> list[Case]:
    g = rng(407)
    out: list[Case] = []
    t = np.sort(g.uniform(0, 20, size=40))
    y = 1.5 * np.sin(1.7 * t + 0.4) + 0.8 + 0.3 * g.normal(size=40)
    freqs = np.linspace(0.2, 4.0, 25)
    w = g.uniform(0.5, 2.0, size=40)
    for floating in [False, True]:
        for norm in ["power", "normalize"]:
            p = S.lombscargle(t, y, freqs, normalize=norm == "normalize", floating_mean=floating)
            out.append(
                case(
                    f"lombScargle/{norm}/{floating}",
                    "lombScargle",
                    [t, y, freqs],
                    p,
                    **TOL,
                    normalize=norm,
                    floatingMean=floating,
                )
            )
        p = S.lombscargle(t, y, freqs, weights=w, normalize=True, floating_mean=floating)
        out.append(
            case(
                f"lombScargle/weights/{floating}",
                "lombScargleWeighted",
                [t, y, freqs, w],
                p,
                **TOL,
                normalize="normalize",
                floatingMean=floating,
            )
        )
        a = S.lombscargle(t, y, freqs, normalize="amplitude", floating_mean=floating)
        out.append(
            case(
                f"lombScargleAmplitude/{floating}",
                "lombScargleAmplitude",
                [t, y, freqs],
                pair(a),
                **TOL,
                floatingMean=floating,
            )
        )
    # The statistics module directly: the class rounds fmax to its automatic frequency grid.
    z = np.array([0.05, 0.2, 0.4, 0.6])
    for method in METHODS:
        fap = A.false_alarm_probability(z, 0.8, t, y, None, normalization="standard", method=method)
        out.append(
            case(
                f"fap/{method}",
                "falseAlarmProbability",
                [t, z],
                np.asarray(fap),
                rtol=1e-9,
                atol=1e-14,
                fmax=0.8,
                method=method,
            )
        )
    for method in ["naive", "baluev"]:
        level = A.false_alarm_level(np.array([0.1, 0.01]), 0.8, t, y, None, normalization="standard", method=method)
        out.append(
            case(
                f"fal/{method}",
                "falseAlarmLevel",
                [t, np.array([0.1, 0.01])],
                np.asarray(level),
                rtol=1e-6,
                atol=1e-9,
                fmax=0.8,
                method=method,
            )
        )
    return out
