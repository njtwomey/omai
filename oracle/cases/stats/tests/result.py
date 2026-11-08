"""Equal-variance tests shared as assumption checks (stats/tests/result.ts) against scipy.stats."""

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(404)
    a = g.normal(0.0, 1.0, size=12)
    b = g.normal(0.3, 2.0, size=15)
    c = g.normal(-0.2, 1.4, size=9)
    out: list[Case] = []
    for name, f in [
        ("levene/median", lambda: st.levene(a, b, c)),
        ("levene/mean", lambda: st.levene(a, b, c, center="mean")),
        ("bartlett", lambda: st.bartlett(a, b, c)),
        ("fligner", lambda: st.fligner(a, b, c)),
        ("fligner/mean", lambda: st.fligner(a, b, c, center="mean")),
    ]:
        r = f()
        center = "mean" if name.endswith("mean") else "median"
        out.append(
            case(name, name.split("/")[0], [a, b, c], np.asarray([r.statistic, r.pvalue]), rtol=1e-10, center=center)
        )
    return out
