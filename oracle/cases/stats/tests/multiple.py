"""Multiple testing against statsmodels.stats.multitest.multipletests and scipy.stats.combine_pvalues."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.stats.multitest import multipletests

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

METHODS = {
    "bonferroni": "bonferroni",
    "sidak": "sidak",
    "holm": "holm",
    "holm-sidak": "holm-sidak",
    "hochberg": "simes-hochberg",
    "hommel": "hommel",
    "benjamini-hochberg": "fdr_bh",
    "benjamini-yekutieli": "fdr_by",
}
Combine = Literal["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]
COMBINE: list[Combine] = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]


def cases() -> list[Case]:
    g = rng(1)
    p = np.concatenate([g.uniform(0, 0.05, 6), g.uniform(0, 1, 14)])
    g.shuffle(p)
    out: list[Case] = []
    for ours, theirs in METHODS.items():
        reject, corrected, _, _ = multipletests(p, alpha=0.1, method=theirs)
        out.append(case(f"adjust/{ours}", "adjust", [p, p], corrected, method=ours))
        out.append(case(f"reject/{ours}", "reject", [p, p], reject.astype(float), method=ours))
    q = g.uniform(0.01, 0.6, 7)
    w = g.uniform(0.5, 2.0, 7)
    for method in COMBINE:
        r = st.combine_pvalues(q, method=method)
        out.append(
            case(
                f"combine/{method}",
                "combine",
                [q, q],
                np.asarray([r.statistic, r.pvalue]),
                method=method.replace("_", "-"),
            )
        )
    r = st.combine_pvalues(q, method="stouffer", weights=w)
    out.append(case("combine/stouffer/weighted", "combineWeighted", [q, w], np.asarray([r.statistic, r.pvalue])))
    return out
