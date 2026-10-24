"""Two-sample tests of equal distributions against scipy.stats."""

import warnings
from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]


def cases() -> list[Case]:
    g = rng(2020)
    out: list[Case] = []
    pairs = {
        "equal": (g.normal(0, 1, 15), g.normal(0.6, 1, 15)),
        "unequal": (g.normal(0, 1, 12), g.normal(0.5, 1.5, 19)),
        "ties": (np.round(g.normal(0, 1, 14)), np.round(g.normal(0.4, 1, 17))),
    }
    for name, (x, y) in pairs.items():
        for alt in ALTS:
            r = st.ks_2samp(x, y, alternative=alt)
            out.append(
                case(
                    f"ks2samp/{name}/{alt}",
                    "ks2samp",
                    [x, y],
                    np.asarray([r.statistic, r.pvalue]),
                    rtol=1e-9,
                    alternative=alt,
                )
            )
        r = st.ks_2samp(x, y, method="asymp")
        out.append(
            case(
                f"ks2samp/{name}/asymp",
                "ks2samp",
                [x, y],
                np.asarray([r.statistic, r.pvalue]),
                rtol=1e-9,
                method="asymp",
            )
        )
        for method in ["exact", "asymptotic"]:
            r = st.cramervonmises_2samp(x, y, method=method)  # pyright: ignore[reportArgumentType, reportCallIssue]
            out.append(
                case(
                    f"cramervonmises2samp/{name}/{method}",
                    "cramervonmises2samp",
                    [x, y],
                    np.asarray([r.statistic, r.pvalue]),
                    rtol=1e-9,
                    method=method,
                )
            )
        out.append(case(f"energyDistance/{name}", "energyDistance", [x, y], np.asarray(st.energy_distance(x, y))))
        out.append(
            case(
                f"wassersteinDistance/{name}", "wassersteinDistance", [x, y], np.asarray(st.wasserstein_distance(x, y))
            )
        )
        r = st.epps_singleton_2samp(x, y)
        out.append(
            case(f"eppsSingleton/{name}", "eppsSingleton", [x, y], np.asarray([r.statistic, r.pvalue]), rtol=1e-9)
        )
    x, y = pairs["unequal"]
    wx = g.uniform(0.5, 2, x.size)
    wy = g.uniform(0.5, 2, y.size)
    out.append(
        case(
            "energyDistance/weighted",
            "energyDistanceWeighted",
            [x, y, wx, wy],
            np.asarray(st.energy_distance(x, y, wx, wy)),
        )
    )
    out.append(
        case(
            "wassersteinDistance/weighted",
            "wassersteinDistanceWeighted",
            [x, y, wx, wy],
            np.asarray(st.wasserstein_distance(x, y, wx, wy)),
        )
    )
    samples = [g.normal(0, 1, 10), g.normal(0.5, 1, 12), np.round(g.normal(0.2, 1, 9), 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = st.anderson_ksamp(samples, variant="midrank")
    out.append(case("andersonKsamp", "andersonKsamp", samples, np.asarray([r.statistic, r.pvalue]), rtol=1e-9))
    small_x, small_y = g.normal(0, 1, 5), g.normal(1, 1, 5)

    def energy_stat(a: FloatArray, b: FloatArray) -> float:
        return float(st.energy_distance(a, b))

    def wasserstein_stat(a: FloatArray, b: FloatArray) -> float:
        return float(st.wasserstein_distance(a, b))

    r = st.permutation_test((small_x, small_y), energy_stat, alternative="greater")
    out.append(case("energyTest", "energyTest", [small_x, small_y], np.asarray([r.statistic, r.pvalue])))
    r = st.permutation_test((small_x, small_y), wasserstein_stat, alternative="greater")
    out.append(case("wassersteinTest", "wassersteinTest", [small_x, small_y], np.asarray([r.statistic, r.pvalue])))
    return out
