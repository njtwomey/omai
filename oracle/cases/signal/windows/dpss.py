"""Discrete prolate spheroidal sequences against scipy.signal.windows.dpss (tapers, norms, concentration ratios)."""

from typing import Literal

import numpy as np
import scipy
import scipy.signal.windows as W
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for m, nw in [(16, 2.5), (31, 3.0), (64, 4.0), (9, 1.5)]:
        tapers, ratios = W.dpss(m, nw, Kmax=4, return_ratios=True)
        tol = {"rtol": 1e-9, "atol": 1e-11}
        out.append(case(f"dpss/{m}/{nw}/k4", "dpss", [], np.asarray(tapers), **tol, m=m, nw=nw, kmax=4))
        out.append(case(f"dpssRatios/{m}/{nw}", "dpssRatios", [], np.asarray(ratios), **tol, m=m, nw=nw, kmax=4))
        norms: list[Literal["approximate", "subsample"]] = ["approximate", "subsample"]
        for norm in norms:
            w = np.asarray(W.dpss(m, nw, norm=norm))
            out.append(case(f"dpss/{m}/{nw}/{norm}", "dpss", [], w, **tol, m=m, nw=nw, norm=norm))
        wp = np.asarray(W.dpss(m, nw, Kmax=2, sym=False))
        out.append(case(f"dpss/{m}/{nw}/periodic", "dpss", [], wp, **tol, m=m, nw=nw, kmax=2, sym=False))
    return out
