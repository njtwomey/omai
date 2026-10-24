"""Robust statistics against scipy.stats (trim_mean, trimboth, mstats.winsorize) and statsmodels (RLM with Huber's
and Tukey's norms on a constant, MAD scale)."""

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
from scipy.stats import mstats
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(66)
    x = np.concatenate([g.normal(5, 1, 40), [25.0, -12.0, 30.0]])
    out: list[Case] = []
    for p in [0.1, 0.2]:
        out.append(case(f"trimMean/{p}", "trimMean", [x, x], np.asarray(st.trim_mean(x, p)), p=p))
        out.append(case(f"trimboth/{p}", "trimboth", [x, x], np.sort(st.trimboth(x, p)), p=p))
        w = np.asarray(mstats.winsorize(x, limits=(p, p / 2)))
        out.append(case(f"winsorize/{p}", "winsorize", [x, x], w.astype(float), lo=p, hi=p / 2))
        out.append(
            case(f"winsorizedVariance/{p}", "winsorizedVariance", [x, x], np.asarray(w.var(ddof=1)), lo=p, hi=p / 2)
        )
    for name, norm in [("huber", sm.robust.norms.HuberT()), ("tukey", sm.robust.norms.TukeyBiweight())]:
        r = sm.RLM(x, np.ones((x.size, 1)), M=norm).fit(tol=1e-14, maxiter=1000)
        out.append(
            case(f"mEstimate/{name}", "mEstimate", [x, x], np.asarray([r.params[0], r.scale]), rtol=1e-8, norm=name)
        )
    return out
