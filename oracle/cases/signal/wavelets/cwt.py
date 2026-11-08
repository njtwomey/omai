"""Continuous wavelet transforms against pywt.cwt, wavelet functions against pywt.ContinuousWavelet.wavefun and
centre frequencies against pywt.central_frequency."""

import numpy as np
import pywt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.signal_helpers import pair
from pywt._extensions._pywt import ContinuousWavelet  # pywt re-exports it untyped

REFERENCE: dict[str, str] = {"pywt": pywt.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-11}

NAMES = ["morl", "mexh", "gaus1", "gaus2", "gaus5", "gaus8", "cmor1.5-1.0", "shan1.5-1.0"]


def cases() -> list[Case]:
    g = rng(414)
    out: list[Case] = []
    for name in NAMES:
        psi, x = ContinuousWavelet(name).wavefun(6)
        out.append(case(f"wavefun/{name}", "wavefun", [], pair(psi), **TOL, wavelet=name, precision=6))
        out.append(case(f"wavefunX/{name}", "wavefunX", [], np.asarray(x), **TOL, wavelet=name, precision=6))
        cf = pywt.central_frequency(name)
        out.append(case(f"centralFrequency/{name}", "centralFrequency", [], np.array(cf), **TOL, wavelet=name))
    sig = np.cos(np.linspace(0, 12, 40) ** 1.5) + 0.1 * g.normal(size=40)
    scales = np.array([1.0, 2.0, 3.5, 6.0])
    for name in ["morl", "mexh", "gaus3", "cmor1.5-1.0"]:
        c, f = pywt.cwt(sig, scales, name, sampling_period=0.5)
        out.append(case(f"cwt/{name}", "cwt", [sig, scales], pair(c), **TOL, wavelet=name, samplingPeriod=0.5))
        out.append(
            case(f"cwtFreqs/{name}", "cwtFreqs", [sig, scales], np.asarray(f), **TOL, wavelet=name, samplingPeriod=0.5)
        )
    return out
