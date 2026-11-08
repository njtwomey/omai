"""One-way and two-way ANOVA tables against scipy.stats.f_oneway and statsmodels anova_lm (types I, II and III)."""

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from scipy import stats
from statsmodels.formula.api import ols

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def _two_way_data(seed: int, levels_a: int, levels_b: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, levels_a, size=n)
    b = rng.integers(0, levels_b, size=n)
    y = 1.0 + 0.8 * a - 0.5 * b + 0.4 * (a == 1) * (b == 1) + rng.normal(size=n)
    return y, a.astype(float), b.astype(float)


def cases() -> list[Case]:
    out: list[Case] = []
    rng = np.random.default_rng(3)
    groups = [rng.normal(loc=m, size=s) for m, s in [(0.0, 7), (0.6, 9), (1.1, 6)]]
    res = stats.f_oneway(*groups)
    all_y = np.concatenate(groups)
    ss_total = float(((all_y - all_y.mean()) ** 2).sum())
    ss_between = float(sum(len(g) * (g.mean() - all_y.mean()) ** 2 for g in groups))
    out.append(
        case(
            "oneway",
            "oneWay",
            groups,
            np.asarray([res.statistic, res.pvalue, ss_between / ss_total]),
            rtol=1e-10,
        )
    )
    for seed, la, lb, n in [(1, 2, 3, 30), (2, 3, 3, 41)]:
        y, a, b = _two_way_data(seed, la, lb, n)
        df = pd.DataFrame({"y": y, "a": a, "b": b})
        model = ols("y ~ C(a, Sum) * C(b, Sum)", data=df).fit()
        for ss_type, typ in [("I", 1), ("II", 2), ("III", 3)]:
            table = sm.stats.anova_lm(model, typ=typ)
            rows = ["C(a, Sum)", "C(b, Sum)", "C(a, Sum):C(b, Sum)"]
            ss = [float(table.loc[r, "sum_sq"]) for r in rows]
            f = [float(table.loc[r, "F"]) for r in rows]
            p = [float(table.loc[r, "PR(>F)"]) for r in rows]
            resid = float(table.loc["Residual", "sum_sq"])
            out.append(
                case(
                    f"twoway/{seed}/type{ss_type}",
                    "twoWay",
                    [y, a, b],
                    np.asarray([*ss, *f, *p, resid]),
                    rtol=1e-8,
                    atol=1e-10,
                    ssType=ss_type,
                )
            )
    return out
