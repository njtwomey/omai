"""Proportion tests and intervals against scipy.stats.binomtest and statsmodels.stats.proportion."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
    test_proportions_2indep,
)
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]
SM = {"two-sided": "two-sided", "less": "smaller", "greater": "larger"}
INTERVALS = {
    "wald": "normal",
    "wilson": "wilson",
    "agresti-coull": "agresti_coull",
    "clopper-pearson": "beta",
    "jeffreys": "jeffreys",
}


def cases() -> list[Case]:
    out: list[Case] = []
    for k, n, p in [(7, 20, 0.5), (3, 40, 0.2), (18, 25, 0.6), (0, 12, 0.3), (12, 12, 0.7)]:
        for alt in ALTS:
            r = st.binomtest(k, n, p, alternative=alt)
            out.append(
                case(f"binomTest/{k}-{n}/{alt}", "binomTest", [], np.asarray(r.pvalue), k=k, n=n, p=p, alternative=alt)
            )
        ci = st.binomtest(k, n, p).proportion_ci(0.9, method="exact")
        out.append(
            case(f"binomTest/ci/{k}-{n}", "binomTestCi", [], np.asarray([ci.low, ci.high]), rtol=1e-10, k=k, n=n, p=p)
        )
        for ours, theirs in INTERVALS.items():
            lo, hi = proportion_confint(k, n, alpha=0.1, method=theirs)
            out.append(
                case(
                    f"proportionConfint/{ours}/{k}-{n}",
                    "proportionConfint",
                    [],
                    np.asarray([lo, hi]),
                    rtol=1e-10,
                    k=k,
                    n=n,
                    method=ours,
                )
            )
    for alt in ALTS:
        z, p = proportions_ztest(14, 40, value=0.5, alternative=SM[alt])
        out.append(case(f"proportionsZtest/one/{alt}", "proportionsZtest1", [], np.asarray([z, p]), alternative=alt))
        z, p = proportions_ztest(np.array([30, 21]), np.array([60, 70]), alternative=SM[alt])
        out.append(case(f"proportionsZtest/two/{alt}", "proportionsZtest2", [], np.asarray([z, p]), alternative=alt))
    for method in ["wald", "agresti-caffo", "newcombe"]:
        theirs = "newcomb" if method == "newcombe" else method
        lo, hi = confint_proportions_2indep(30, 60, 21, 70, method=theirs, alpha=0.1)
        out.append(
            case(f"proportionsDifference/{method}", "proportionsDifferenceCi", [], np.asarray([lo, hi]), method=method)
        )
    t = test_proportions_2indep(30, 60, 21, 70, method="wald")
    out.append(case("proportionsDifference/test", "proportionsDifferenceTest", [], np.asarray([t.statistic, t.pvalue])))  # pyright: ignore[reportAttributeAccessIssue]
    return out
