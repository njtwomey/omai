"""Cosine-sum windows against scipy.signal.windows, symmetric and periodic."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.signal.windows as W
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

WINDOWS: dict[str, Callable[..., FloatArray]] = {
    "boxcar": W.boxcar,
    "hann": W.hann,
    "hamming": W.hamming,
    "blackman": W.blackman,
    "blackmanHarris": W.blackmanharris,
    "nuttall": W.nuttall,
    "flattop": W.flattop,
}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, f in WINDOWS.items():
        for m in [1, 2, 7, 16]:
            for sym in [True, False]:
                out.append(case(f"{name}/{m}/{sym}", name, [], np.asarray(f(m, sym=sym)), m=m, sym=sym))
    for m in [5, 12]:
        a = [0.4, 0.4, 0.2]
        out.append(case(f"generalCosine/{m}", "generalCosine", [], W.general_cosine(m, a), m=m, a=a))
        w = W.general_hamming(m, 0.6, sym=False)
        out.append(case(f"generalHamming/{m}", "generalHamming", [], w, m=m, alpha=0.6, sym=False))
    return out
