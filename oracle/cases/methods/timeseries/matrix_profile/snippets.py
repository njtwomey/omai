"""MPdist, time-series snippets and the consensus motif (ostinato) against STUMPY."""

from importlib.metadata import version

import numpy as np
import stumpy
from omai_oracle.fixture import Case, case
from omai_oracle.timeseries_helpers import as_float, series

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-8}
# Exact repeats (a query taken from the series, copied dimensions, planted motifs) have distances sqrt(2m(1 - rho))
# with rho within rounding of 1, where 1e-14 in rho is 1e-7 in the distance: those cases get atol 1e-6.
SELF = {"rtol": 1e-7, "atol": 1e-6}


def cases() -> list[Case]:
    out: list[Case] = []
    g = np.random.default_rng(625)
    seg = np.concatenate([np.sin(np.arange(150) / 3.0), np.sign(np.sin(np.arange(150) / 5.0))])
    seg = seg + 0.05 * g.normal(size=300)
    a = series(70, 626)
    b = series(55, 627)
    out.append(case("mpdist", "mpdist", [a, b], np.atleast_1d(stumpy.mpdist(a, b, 8)), **TOL, m=8))
    out.append(case("mpdist/k", "mpdist", [a, b], np.atleast_1d(stumpy.mpdist(a, b, 8, k=10)), **TOL, m=8, k=10))
    sn = seg[::3][:100].copy()
    snips, idx, profiles, fractions, areas, regimes = stumpy.snippets(sn, 10, 2)
    out.append(
        case(
            "snippets",
            "snippets",
            [sn],
            np.concatenate(
                [as_float(idx), as_float(profiles), as_float(fractions), as_float(areas), as_float(regimes)]
            ),
            rtol=1e-7,
            atol=1e-7,
            m=10,
            k=2,
        )
    )
    del snips
    snips, idx, profiles, fractions, areas, regimes = stumpy.snippets(sn, 10, 2, s=5)
    out.append(
        case(
            "snippets/s5",
            "snippets",
            [sn],
            np.concatenate(
                [as_float(idx), as_float(profiles), as_float(fractions), as_float(areas), as_float(regimes)]
            ),
            rtol=1e-7,
            atol=1e-7,
            m=10,
            k=2,
            s=5,
        )
    )
    ts = [series(50, 630), series(60, 631), series(45, 632)]
    motif = np.sin(np.linspace(0, 2 * np.pi, 8)) * 2
    noise = np.random.default_rng(634).normal(size=(3, 8)) * 0.1
    ts[0][10:18] = motif + noise[0]
    ts[1][30:38] = motif + noise[1]
    ts[2][5:13] = motif + noise[2]
    r, si, ii = stumpy.ostinato(ts, 8)
    out.append(case("ostinato", "ostinato", ts, np.array([r, float(si), float(ii)]), **SELF, m=8))
    return out
