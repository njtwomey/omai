"""Frequency responses against scipy.signal: sosfreqz, freqz_zpk and group_delay."""

from typing import Any

import numpy as np
import scipy
import scipy.signal as S
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.signal_helpers import pair

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    out: list[Case] = []
    sos = S.ellip(6, 1.0, 50.0, 0.3, output="sos")
    _, h = S.sosfreqz(sos, worN=16)
    out.append(case("sosfreqz/count", "sosfreqz", [sos], pair(h), **TOL, worN=16))
    f = np.array([10.0, 120.0, 250.0, 400.0])
    _, h = S.sosfreqz(sos, worN=f, fs=1000.0)
    out.append(case("sosfreqz/fs", "sosfreqzAt", [sos, f], pair(h), **TOL, fs=1000.0))
    z, p, k = S.cheby2(5, 40.0, 0.4, output="zpk")
    _, h = S.freqz_zpk(z, p, k, worN=12, whole=True)
    zpk: dict[str, Any] = {"zr": z.real, "zi": z.imag, "pr": p.real, "pi": p.imag, "k": float(k)}
    out.append(case("freqzZpk", "freqzZpk", [], pair(h), **TOL, worN=12, whole=True, **zpk))
    b, a = S.butter(4, 0.25)
    _, gd = S.group_delay((b, a), w=16)
    out.append(case("groupDelay/iir", "groupDelay", [b, a], gd, rtol=1e-9, atol=1e-10, worN=16))
    fir = S.firwin(21, 0.3)
    _, gd = S.group_delay((fir, [1.0]), w=np.array([0.05, 0.1, 0.2]), fs=1.0)
    out.append(
        case(
            "groupDelay/fir/fs",
            "groupDelayAt",
            [fir, np.array([1.0]), np.array([0.05, 0.1, 0.2])],
            gd,
            rtol=1e-9,
            atol=1e-10,
            fs=1.0,
        )
    )
    return out
