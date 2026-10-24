"""Time-series tests against statsmodels (acorr_ljungbox, durbin_watson, adfuller with MacKinnon's p-values, kpss and
runstest_1samp)."""

import warnings
from typing import Literal

import numpy as np
import statsmodels
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.stattools import adfuller, kpss
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}

Trend = Literal["n", "c", "ct", "ctt"]
TRENDS: list[Trend] = ["n", "c", "ct", "ctt"]


def cases() -> list[Case]:
    g = rng(1111)
    e = g.normal(size=120)
    ar = np.zeros(120)
    for t in range(1, 120):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    walk = np.cumsum(g.normal(size=150)) + 0.05 * np.arange(150)
    out: list[Case] = []
    for lags in [1, 5, 10]:
        lb = acorr_ljungbox(ar, lags=[lags], boxpierce=True)
        out.append(
            case(
                f"ljungBox/{lags}",
                "ljungBox",
                [ar, ar],
                np.asarray([lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]]),
                lags=lags,
            )
        )
        out.append(
            case(
                f"boxPierce/{lags}",
                "boxPierce",
                [ar, ar],
                np.asarray([lb["bp_stat"].iloc[0], lb["bp_pvalue"].iloc[0]]),
                lags=lags,
            )
        )
    out.append(case("durbinWatson", "durbinWatson", [e, e], np.asarray(durbin_watson(e))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for series, data in [("stationary", ar), ("walk", walk)]:
            for reg in TRENDS:
                for autolag in ["AIC", "BIC", "t-stat"]:
                    r = adfuller(data, regression=reg, autolag=autolag, result_object=True)  # pyright: ignore[reportArgumentType, reportCallIssue]
                    out.append(
                        case(
                            f"adfuller/{series}/{reg}/{autolag}",
                            "adfuller",
                            [data, data],
                            np.asarray([r.statistic, r.pvalue, r.lags]),  # pyright: ignore[reportAttributeAccessIssue]
                            rtol=1e-9,
                            regression=reg,
                            autolag=autolag,
                        )
                    )
                r = adfuller(data, regression=reg, autolag=None, maxlag=3, result_object=True)  # pyright: ignore[reportArgumentType, reportCallIssue]
                out.append(
                    case(
                        f"adfuller/{series}/{reg}/fixed",
                        "adfuller",
                        [data, data],
                        np.asarray([r.statistic, r.pvalue, r.lags]),  # pyright: ignore[reportAttributeAccessIssue]
                        rtol=1e-9,
                        regression=reg,
                        maxlag=3,
                    )
                )
            for reg in ["c", "ct"]:
                for nlags in ["auto", "legacy", 4]:
                    stat, p, lags, _ = kpss(data, regression=reg, nlags=nlags)  # pyright: ignore[reportArgumentType]
                    out.append(
                        case(
                            f"kpss/{series}/{reg}/{nlags}",
                            "kpss",
                            [data, data],
                            np.asarray([stat, p, lags]),
                            regression=reg,
                            nlags=nlags,
                        )
                    )
    for stat in [-4.2, -3.1, -2.0, -1.0, 0.3]:
        for reg in TRENDS:
            out.append(
                case(
                    f"mackinnonp/{reg}/{stat}",
                    "mackinnonp",
                    [],
                    np.asarray(mackinnonp(stat, regression=reg)),
                    stat=stat,
                    regression=reg,
                )
            )
    for reg in TRENDS:
        out.append(
            case(
                f"mackinnoncrit/{reg}",
                "mackinnoncrit",
                [],
                np.asarray(mackinnoncrit(regression=reg, nobs=80)),
                regression=reg,
            )
        )
    for cutoff in ["mean", "median"]:
        for corr in [True, False]:
            z, p = runstest_1samp(e[:40], cutoff=cutoff, correction=corr)
            out.append(
                case(
                    f"runsTest/{cutoff}/{corr}",
                    "runsTest",
                    [e[:40], e[:40]],
                    np.asarray([z, p]),
                    cutoff=cutoff,
                    correction=corr,
                )
            )
    return out
