"""The DTFT against scipy.signal.freqz, and the analytic signal against scipy.signal.hilbert."""

import numpy as np
import scipy
import scipy.signal as S
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import pair, tones

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(401)
    out: list[Case] = []
    x = g.normal(size=12)
    f = np.array([0.0, 0.05, 0.13, 0.25, 0.4, 0.5])
    _, h = S.freqz(x, worN=f, fs=1.0)
    out.append(case("dtft/real", "dtft", [x, f], pair(h), rtol=1e-11, atol=1e-11, fs=1.0))
    _, h2 = S.freqz(x, worN=f * 8, fs=8.0)
    out.append(case("dtft/fs", "dtft", [x, f * 8], pair(h2), rtol=1e-11, atol=1e-11, fs=8.0))
    for n in [8, 9, 33]:
        s = tones(g, n)
        out.append(case(f"hilbert/{n}", "hilbert", [s], pair(S.hilbert(s)), rtol=1e-11, atol=1e-12))
    s = tones(g, 10)
    out.append(case("hilbert/padded", "hilbert", [s], pair(S.hilbert(s, N=16)), rtol=1e-11, atol=1e-12, n=16))
    m = g.normal(size=(3, 7))
    out.append(case("hilbert/axis0", "hilbert", [m], pair(S.hilbert(m, axis=0)), rtol=1e-11, atol=1e-12, axis=0))
    return out
