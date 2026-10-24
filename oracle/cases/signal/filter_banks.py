"""Filter banks: pywt.qmf (scipy.signal.qmf was removed in scipy 1.18), and wavelet packets against
pywt.WaveletPacket (nodes in natural and frequency order, reconstruction)."""

from typing import Any

import numpy as np
import pywt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"pywt": pywt.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(416)
    out: list[Case] = []
    h = g.normal(size=7)
    out.append(case("qmf", "qmf", [h], pywt.qmf(h), **TOL))
    x = g.normal(size=37)
    for name, mode, level in [("db2", "symmetric", 3), ("haar", "periodization", 3), ("sym4", "zero", 2)]:
        wp = pywt.WaveletPacket(x, name, mode=mode, maxlevel=level)
        for order in ["natural", "freq"]:
            nodes = wp.get_level(level, order)
            flat = np.concatenate([np.asarray(n.data) for n in nodes])
            params: dict[str, Any] = {"wavelet": name, "mode": mode, "level": level, "order": order}
            out.append(case(f"packet/{name}/{mode}/{order}", "packetNodes", [x], flat, **TOL, **params))
        out.append(
            case(
                f"packetRec/{name}/{mode}",
                "packetReconstruct",
                [x],
                wp.reconstruct(update=False),
                **TOL,
                wavelet=name,
                mode=mode,
                level=level,
            )
        )
    return out
