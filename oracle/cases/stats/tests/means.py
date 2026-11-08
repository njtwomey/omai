"""Tests of means against scipy.stats (t tests, one-way ANOVA, Tukey's HSD), statsmodels (z tests) and pingouin
(Welch's ANOVA, repeated-measures ANOVA with sphericity)."""

from typing import Literal

import numpy as np
import pandas as pd
import pingouin as pg
import scipy
import scipy.stats as st
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from statsmodels.stats.weightstats import ztest

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "pingouin": pg.__version__,
}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]
SM = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}


def ttest_row(stat: float, p: float, df: object, lo: float, hi: float) -> FloatArray:
    return np.asarray([stat, p, float(np.asarray(df)), lo, hi])


def cases() -> list[Case]:
    g = rng(99)
    x = g.normal(0.4, 1.0, size=14)
    y = g.normal(0.0, 1.8, size=11)
    pre = g.normal(10, 2, size=12)
    post = pre + g.normal(0.6, 1.0, size=12)
    out: list[Case] = []
    for alt in ALTS:
        r = st.ttest_1samp(x, 0.1, alternative=alt)
        ci = r.confidence_interval(0.9)
        out.append(
            case(
                f"ttest1samp/{alt}",
                "ttest1samp",
                [x, x],
                ttest_row(r.statistic, r.pvalue, r.df, ci.low, ci.high),
                popmean=0.1,
                alternative=alt,
            )
        )
        for equal in [True, False]:
            r = st.ttest_ind(x, y, equal_var=equal, alternative=alt)
            ci = r.confidence_interval(0.9)
            out.append(
                case(
                    f"ttestInd/{'student' if equal else 'welch'}/{alt}",
                    "ttestInd",
                    [x, y],
                    ttest_row(r.statistic, r.pvalue, r.df, ci.low, ci.high),
                    rtol=1e-10,
                    equalVar=equal,
                    alternative=alt,
                )
            )
        r = st.ttest_rel(post, pre, alternative=alt)
        ci = r.confidence_interval(0.9)
        out.append(
            case(
                f"ttestRel/{alt}",
                "ttestRel",
                [post, pre],
                ttest_row(r.statistic, r.pvalue, r.df, ci.low, ci.high),
                alternative=alt,
            )
        )
        z, p = ztest(x, value=0.1, alternative=SM[alt])  # pyright: ignore[reportArgumentType]
        out.append(case(f"ztest/one/{alt}", "ztest1", [x, x], np.asarray([z, p]), value=0.1, alternative=alt))
        for usevar in ["pooled", "unequal"]:
            z, p = ztest(x, y, value=0.2, alternative=SM[alt], usevar=usevar)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"ztest/two/{usevar}/{alt}",
                    "ztest2",
                    [x, y],
                    np.asarray([z, p]),
                    value=0.2,
                    alternative=alt,
                    usevar=usevar,
                )
            )
    a, b, c = g.normal(0, 1, 10), g.normal(0.8, 1, 12), g.normal(0.3, 2, 9)
    r = st.f_oneway(a, b, c)
    out.append(case("fOneway", "fOneway", [a, b, c], np.asarray([r.statistic, r.pvalue])))
    frame = pd.DataFrame({"y": np.concatenate([a, b, c]), "g": np.repeat(["a", "b", "c"], [a.size, b.size, c.size])})
    w = pg.welch_anova(data=frame, dv="y", between="g")
    out.append(
        case(
            "welchAnova",
            "welchAnova",
            [a, b, c],
            np.asarray([w["F"].iloc[0], w["p_unc"].iloc[0], w["ddof2"].iloc[0], w["np2"].iloc[0]]),
            rtol=1e-10,
        )
    )
    t = st.tukey_hsd(a, b, c)
    ci = t.confidence_interval(0.95)
    for name, value in [("statistic", t.statistic), ("pvalue", t.pvalue), ("low", ci.low), ("high", ci.high)]:
        # scipy's studentized range integrates adaptively; ours uses a fixed product rule.
        out.append(case(f"tukeyHsd/{name}", "tukeyHsd", [a, b, c], np.asarray(value), rtol=1e-7, atol=1e-10, part=name))
    subjects = 10
    wide = (
        g.normal(size=(subjects, 1))
        + g.normal(size=(subjects, 4)) * np.array([1.0, 1.5, 0.7, 2.0])
        + [0, 0.3, 0.6, 0.2]
    )
    long = pd.DataFrame(
        {
            "s": np.repeat(np.arange(subjects), 4),
            "w": np.tile(["c1", "c2", "c3", "c4"], subjects),
            "y": wide.ravel(),
        }
    )
    for effsize in ["ng2", "np2"]:
        aov = pg.rm_anova(data=long, dv="y", within="w", subject="s", correction=True, effsize=effsize)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"rmAnova/{effsize}",
                "rmAnova",
                [wide, wide],
                np.asarray(
                    [
                        aov["F"].iloc[0],
                        aov["p_unc"].iloc[0],
                        aov[effsize].iloc[0],
                        aov["eps"].iloc[0],
                        aov["p_GG_corr"].iloc[0],
                        aov["W_spher"].iloc[0],
                        aov["p_spher"].iloc[0],
                    ]
                ),
                rtol=1e-9,
                effsize=effsize,
            )
        )
    return out
