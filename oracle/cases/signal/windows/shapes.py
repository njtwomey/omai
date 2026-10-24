"""Other windows against scipy.signal.windows: triangular, Kaiser, Gaussian, Tukey and the rest."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.signal.windows as W
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

PLAIN: dict[str, Callable[..., FloatArray]] = {
    "bartlett": W.bartlett,
    "triang": W.triang,
    "cosine": W.cosine,
    "parzen": W.parzen,
    "bohman": W.bohman,
    "barthann": W.barthann,
    "lanczos": W.lanczos,
}


def both(out: list[Case], name: str, m: int, w: FloatArray, wp: FloatArray, **params: float) -> None:
    """Symmetric and periodic cases of one window."""
    out.append(case(f"{name}/{m}/sym", name, [], w, rtol=1e-12, atol=1e-14, m=m, sym=True, **params))
    out.append(case(f"{name}/{m}/periodic", name, [], wp, rtol=1e-12, atol=1e-14, m=m, sym=False, **params))


def cases() -> list[Case]:
    out: list[Case] = []
    for m in [1, 2, 7, 16]:
        for name, f in PLAIN.items():
            both(out, name, m, f(m), f(m, sym=False))
        both(out, "kaiser", m, W.kaiser(m, 6.5), W.kaiser(m, 6.5, sym=False), beta=6.5)
        both(out, "gaussian", m, W.gaussian(m, 2.5), W.gaussian(m, 2.5, sym=False), std=2.5)
        gg = W.general_gaussian(m, 1.5, 3.0)
        both(out, "generalGaussian", m, gg, W.general_gaussian(m, 1.5, 3.0, sym=False), p=1.5, sig=3.0)
        both(out, "tukey", m, W.tukey(m, 0.3), W.tukey(m, 0.3, sym=False), alpha=0.3)
        both(out, "exponential", m, W.exponential(m, tau=2.0), W.exponential(m, tau=2.0, sym=False), tau=2.0)
    centred = W.exponential(9, center=2.0, tau=3.0, sym=False)
    out.append(case("exponential/center", "exponential", [], centred, m=9, sym=False, tau=3.0, center=2.0))
    for alpha in [0.0, 1.0, 0.75]:
        out.append(case(f"tukey/alpha{alpha}", "tukey", [], W.tukey(10, alpha), m=10, sym=True, alpha=alpha))
    for m in [4, 10, 16]:
        kbd = np.asarray(W.kaiser_bessel_derived(m, 4.0))
        out.append(case(f"kaiserBesselDerived/{m}", "kaiserBesselDerived", [], kbd, m=m, beta=4.0))
    return out
