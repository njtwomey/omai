"""Filter frequency responses (freqs and freqz, including the FFT path for FIR filters) against scipy.signal."""

import numpy as np
import scipy
import scipy.signal as sg
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.lti import parts

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

TOL: dict[str, float] = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    out: list[Case] = []
    butter_b, butter_a = sg.butter(3, 2.0, analog=True)
    analog: list[tuple[str, list[float], list[float]]] = [
        ("butter", [float(v) for v in butter_b], [float(v) for v in butter_a]),
        ("lead", [1.0, 0.5], [1.0, 5.0]),
        ("integrator", [1.0], [1.0, 1.0, 0.0]),
        ("resonant", [1.0, 0.0, 4.0], [1.0, 0.2, 9.0, 0.3]),
    ]
    for name, bb, aa in analog:
        wf, h = sg.freqs(bb, aa, worN=24)
        out.append(case(f"freqs/{name}/w", "freqs-w", [np.array(bb), np.array(aa)], wf, rtol=1e-12, n=24))
        out.append(case(f"freqs/{name}/h", "freqs-h", [np.array(bb), np.array(aa)], parts(h), **TOL, n=24))
    fir = [0.1, 0.25, 0.3, 0.25, 0.1]
    iir_b, iir_a = sg.butter(4, 0.3)
    fz: list[tuple[str, list[float], list[float], dict[str, object]]] = [
        ("fir-fft", fir, [1.0], {"worN": 8}),
        ("fir-fft-whole", fir, [2.0], {"worN": 16, "whole": True}),
        ("fir-nyquist", fir, [1.0], {"worN": 9, "include_nyquist": True}),
        ("iir", list(iir_b), list(iir_a), {"worN": 10}),
        ("iir-fs", list(iir_b), list(iir_a), {"worN": 6, "fs": 1000.0, "whole": True}),
    ]
    for name, bb, aa, kw in fz:
        wz, h = sg.freqz(bb, aa, **kw)  # type: ignore[arg-type]
        params = {k: v for k, v in kw.items() if isinstance(v, int | float | bool)}
        out.append(case(f"freqz/{name}/w", "freqz-w", [np.array(bb), np.array(aa)], wz, rtol=1e-12, **params))
        out.append(case(f"freqz/{name}/h", "freqz-h", [np.array(bb), np.array(aa)], parts(h), **TOL, **params))
    wq = np.linspace(0, np.pi, 7)
    _, h = sg.freqz(iir_b, iir_a, worN=wq)
    out.append(case("freqz/given", "freqz-given", [np.array(iir_b), np.array(iir_a), wq], parts(h), **TOL))
    return out
