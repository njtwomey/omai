"""FA*IR's table of minimum protected candidates against scipy's binomial quantiles."""

import numpy as np
import scipy
import scipy.stats as S
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for k, p, alpha in [(20, 0.5, 0.1), (30, 0.3, 0.05), (15, 0.7, 0.1)]:
        table = np.asarray([S.binom.ppf(alpha, i, p) for i in range(1, k + 1)], dtype=np.float64)
        out.append(case(f"fairMinimumTable/{k}-{p}-{alpha}", "mtable", [], table, k=k, p=p, alpha=alpha))
    return out
