"""Covariance estimators against sklearn.covariance."""

import numpy as np
import sklearn
from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance, empirical_covariance
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(31415)
    X = g.normal(size=(25, 4)) @ g.normal(size=(4, 4)) + np.array([1.0, -2.0, 0.5, 3.0])
    wide = g.normal(size=(8, 12))
    out: list[Case] = []
    for name, data in [("tall", X), ("wide", wide)]:
        for centred in [False, True]:
            tag = f"{name}/{'centred' if centred else 'raw'}"
            lw = LedoitWolf(assume_centered=centred).fit(data)
            oa = OAS(assume_centered=centred).fit(data)
            sh = ShrunkCovariance(assume_centered=centred, shrinkage=0.3).fit(data)
            out += [
                case(
                    f"empirical/{tag}",
                    "empirical",
                    [data, data],
                    empirical_covariance(data, assume_centered=centred),
                    assumeCentered=centred,
                ),
                case(
                    f"ledoitWolf/{tag}", "ledoitWolf", [data, data], lw.covariance_, rtol=1e-10, assumeCentered=centred
                ),
                case(
                    f"ledoitWolf/shrinkage/{tag}",
                    "ledoitWolfShrinkage",
                    [data, data],
                    np.asarray(lw.shrinkage_),
                    rtol=1e-10,
                    assumeCentered=centred,
                ),
                case(f"oas/{tag}", "oas", [data, data], oa.covariance_, rtol=1e-10, assumeCentered=centred),
                case(
                    f"oas/shrinkage/{tag}",
                    "oasShrinkage",
                    [data, data],
                    np.asarray(oa.shrinkage_),
                    rtol=1e-10,
                    assumeCentered=centred,
                ),
                case(f"shrunk/{tag}", "shrunk", [data, data], sh.covariance_, assumeCentered=centred),
            ]
    return out
