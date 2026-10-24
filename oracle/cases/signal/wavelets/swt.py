"""Stationary wavelet transforms against pywt.swt and pywt.iswt."""

import numpy as np
import pywt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"pywt": pywt.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(413)
    out: list[Case] = []
    x = g.normal(size=32)
    for name, level in [("haar", 3), ("db2", 2), ("sym4", 3), ("bior2.2", 2), ("coif1", 1)]:
        coeffs = pywt.swt(x, name, level=level)
        flat = np.concatenate([np.concatenate([a, d]) for a, d in coeffs])
        out.append(case(f"swt/{name}/{level}", "swt", [x], flat, **TOL, wavelet=name, level=level))
        out.append(case(f"iswt/{name}/{level}", "iswt", [x], pywt.iswt(coeffs, name), **TOL, wavelet=name, level=level))
    coeffs = pywt.swt(x, "db2", level=2, start_level=1)
    flat = np.concatenate([np.concatenate([a, d]) for a, d in coeffs])
    out.append(case("swt/start", "swt", [x], flat, **TOL, wavelet="db2", level=2, startLevel=1))
    return out
