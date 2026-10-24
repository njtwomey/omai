"""Equivalence (TOST) tests against statsmodels (ttost_ind, ttost_paired, DescrStatsW.ttost_mean, binom_tost,
tost_proportions_2indep)."""

import numpy as np
import statsmodels
from statsmodels.stats.proportion import binom_tost, tost_proportions_2indep
from statsmodels.stats.weightstats import DescrStatsW, ttost_ind, ttost_paired
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(31)
    x = g.normal(0.1, 1, 30)
    y = g.normal(0.0, 1.3, 26)
    pre = g.normal(5, 1, 20)
    post = pre + g.normal(0.05, 0.4, 20)
    out: list[Case] = []
    for usevar in ["pooled", "unequal"]:
        p, lo, hi = ttost_ind(x, y, -0.6, 0.6, usevar=usevar)
        out.append(
            case(
                f"tost/ind/{usevar}",
                "tostInd",
                [x, y],
                np.asarray([p, lo[0], lo[1], hi[0], hi[1]]),
                equalVar=usevar == "pooled",
            )
        )
    p, lo, hi = ttost_paired(post, pre, -0.3, 0.3)
    out.append(case("tost/paired", "tostPaired", [post, pre], np.asarray([p, lo[0], lo[1], hi[0], hi[1]])))
    p, lo, hi = DescrStatsW(x).ttost_mean(-0.2, 0.5)
    out.append(case("tost/one", "tostOne", [x, x], np.asarray([p, lo[0], lo[1], hi[0], hi[1]])))
    p, p1, p2 = binom_tost(52, 100, 0.4, 0.6)
    out.append(case("tostProportion", "tostProportion", [], np.asarray([p, p1, p2])))
    for method in ["agresti-caffo", "wald"]:
        r = tost_proportions_2indep(60, 120, 58, 115, -0.15, 0.15, method=method)
        out.append(
            case(
                f"tostProportions/{method}",
                "tostProportions",
                [],
                np.asarray([r.statistic, r.pvalue]),  # pyright: ignore[reportAttributeAccessIssue]
                method=method,
            )
        )
    return out
