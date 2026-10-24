"""Contingency-table tests against scipy.stats (chi-square, G, Fisher, odds ratios, relative risk) and statsmodels
(McNemar, Cochran's Q)."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from scipy.stats.contingency import expected_freq, odds_ratio, relative_risk
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]


def cases() -> list[Case]:
    t22 = np.array([[12.0, 5.0], [7.0, 15.0]])
    t23 = np.array([[10.0, 14.0, 6.0], [8.0, 3.0, 12.0]])
    t34 = np.array([[5.0, 9.0, 3.0, 7.0], [8.0, 2.0, 6.0, 4.0], [3.0, 7.0, 9.0, 5.0]])
    out: list[Case] = [case("expectedFrequencies", "expectedFrequencies", [t23, t23], expected_freq(t23))]
    for name, t in [("2x2", t22), ("2x3", t23), ("3x4", t34)]:
        for corr in [True, False]:
            r = st.chi2_contingency(t, correction=corr)
            out.append(
                case(
                    f"chi2Contingency/{name}/{'yates' if corr else 'plain'}",
                    "chi2Contingency",
                    [t, t],
                    np.asarray([r.statistic, r.pvalue]),
                    correction=corr,
                )
            )
        lams: list[Literal["log-likelihood", "freeman-tukey", "neyman", "cressie-read"]] = [
            "log-likelihood",
            "freeman-tukey",
            "neyman",
            "cressie-read",
        ]
        for lam in lams:
            r = st.chi2_contingency(t, correction=False, lambda_=lam)
            out.append(
                case(
                    f"chi2Contingency/{name}/{lam}",
                    "chi2Contingency",
                    [t, t],
                    np.asarray([r.statistic, r.pvalue]),
                    correction=False,
                    lam=lam,
                )
            )
    for t in [t22, np.array([[8.0, 2.0], [1.0, 5.0]]), np.array([[3.0, 0.0], [2.0, 6.0]])]:
        label = "-".join(str(int(v)) for v in t.ravel())
        for alt in ALTS:
            r = st.fisher_exact(t.astype(int), alternative=alt)
            out.append(
                case(
                    f"fisherExact/{label}/{alt}",
                    "fisherExact",
                    [t, t],
                    np.asarray([r.statistic, r.pvalue]),
                    alternative=alt,
                )
            )
    for alt in ALTS:
        res = odds_ratio(t22.astype(int), kind="conditional")
        ci = res.confidence_interval(0.9, alternative=alt)
        out.append(
            case(
                f"oddsRatio/conditional/{alt}",
                "oddsRatio",
                [t22, t22],
                np.asarray([res.statistic, ci.low, ci.high]),
                rtol=1e-8,
                kind="conditional",
                alternative=alt,
            )
        )
        res = odds_ratio(t22.astype(int), kind="sample")
        ci = res.confidence_interval(0.9, alternative=alt)
        out.append(
            case(
                f"oddsRatio/sample/{alt}",
                "oddsRatio",
                [t22, t22],
                np.asarray([res.statistic, ci.low, ci.high]),
                kind="sample",
                alternative=alt,
            )
        )
    rr = relative_risk(20, 80, 12, 90)
    ci = rr.confidence_interval(0.9)
    out.append(case("relativeRisk", "relativeRisk", [], np.asarray([rr.relative_risk, ci.low, ci.high])))
    paired = np.array([[20.0, 9.0], [3.0, 18.0]])
    for exact in [True, False]:
        for corr in [True, False]:
            r = mcnemar(paired, exact=exact, correction=corr)
            out.append(
                case(
                    f"mcnemar/{'exact' if exact else 'chi2'}/{corr}",
                    "mcnemar",
                    [paired, paired],
                    np.asarray([r.statistic, r.pvalue]),  # pyright: ignore[reportAttributeAccessIssue]
                    exact=exact,
                    correction=corr,
                )
            )
    binary = (np.random.default_rng(8).uniform(size=(15, 4)) < np.array([0.3, 0.5, 0.6, 0.8])).astype(float)
    q = cochrans_q(binary)
    out.append(case("cochranQ", "cochranQ", [binary, binary], np.asarray([q.statistic, q.pvalue])))  # pyright: ignore[reportAttributeAccessIssue]
    return out
