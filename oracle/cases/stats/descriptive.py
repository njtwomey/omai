"""Descriptive statistics against numpy, scipy.stats and statsmodels' DescrStatsW (weighted moments)."""

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.stats.weightstats import DescrStatsW

REFERENCE: dict[str, str] = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
}


def cases() -> list[Case]:
    g = rng(1701)
    x = g.gamma(2.0, 1.5, size=25)
    w = g.uniform(0.2, 3.0, size=25)
    small = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    d = st.describe(x)
    weighted = DescrStatsW(x, weights=w, ddof=1)
    out: list[Case] = [
        case("mean", "mean", [x], np.asarray(np.mean(x))),
        case("mean/weighted", "mean", [x, w], np.asarray(np.average(x, weights=w))),
        case("variance", "variance", [x], np.asarray(np.var(x))),
        case("variance/ddof1", "variance", [x], np.asarray(np.var(x, ddof=1)), ddof=1),
        case("variance/weighted", "variance", [x, w], np.asarray(weighted.var), ddof=1),
        case("std/weighted", "std", [x, w], np.asarray(weighted.std), ddof=1),
        case("std", "std", [x], np.asarray(np.std(x, ddof=1)), ddof=1),
        case("skewness", "skewness", [x], np.asarray(st.skew(x))),
        case("skewness/unbiased", "skewness", [x], np.asarray(st.skew(x, bias=False)), bias=False),
        case("kurtosis", "kurtosis", [x], np.asarray(st.kurtosis(x))),
        case("kurtosis/unbiased", "kurtosis", [x], np.asarray(st.kurtosis(x, bias=False)), bias=False),
        case("kurtosis/pearson", "kurtosis", [x], np.asarray(st.kurtosis(x, fisher=False)), fisher=False),
        case("moment/3", "moment", [x], np.asarray(st.moment(x, 3)), order=3),
        case("moment/4", "moment", [x], np.asarray(st.moment(x, 4)), order=4),
        case("moment/about", "moment", [x], np.asarray(st.moment(x, 2, center=1.0)), order=2, center=1.0),
        case(
            "describe",
            "describe",
            [x],
            np.asarray([d.nobs, d.minmax[0], d.minmax[1], d.mean, d.variance, d.skewness, d.kurtosis]),
        ),
        case("zscore", "zscore", [x], np.asarray(st.zscore(x))),
        case("zscore/ddof1", "zscore", [x], np.asarray(st.zscore(x, ddof=1)), ddof=1),
        case("sem", "sem", [x], np.asarray(st.sem(x))),
        case("mode", "mode", [small], np.asarray([st.mode(small).mode, st.mode(small).count])),
        case("geometricMean", "geometricMean", [x], np.asarray(st.gmean(x))),
        case("geometricMean/weighted", "geometricMean", [x, w], np.asarray(st.gmean(x, weights=w))),
        case("harmonicMean", "harmonicMean", [x], np.asarray(st.hmean(x))),
        case("harmonicMean/weighted", "harmonicMean", [x, w], np.asarray(st.hmean(x, weights=w))),
    ]
    return out
