"""Spatial point patterns: Ripley's K with every edge correction of astropy.stats.RipleysKEstimator (none,
translation, Ohser, variable width and Ripley's isotropic correction) on a fixed pattern in a rectangle."""

from typing import Literal

import astropy
import numpy as np
from astropy.stats import RipleysKEstimator
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"astropy": astropy.__version__, "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(31)
    out: list[Case] = []
    pts = np.column_stack([g.uniform(0, 2, size=30), g.uniform(0, 1, size=30)])
    radii = np.array([0.05, 0.1, 0.2, 0.3, 0.45])
    est = RipleysKEstimator(area=2.0, x_min=0, x_max=2, y_min=0, y_max=1)
    window = {"xmin": 0.0, "xmax": 2.0, "ymin": 0.0, "ymax": 1.0}
    modes: list[Literal["none", "translation", "ohser", "var-width", "ripley"]] = [
        "none",
        "translation",
        "ohser",
        "var-width",
        "ripley",
    ]
    for mode in modes:
        k = est(data=pts, radii=radii, mode=mode)
        out.append(case(f"ripley/{mode}", "ripley", [pts, radii], k, rtol=1e-10, atol=1e-12, mode=mode, **window))
    return out
