"""Multivariate tests against pingouin (Hotelling's T-squared, Box's M) and statsmodels (MANOVA). Mardia's tests have
no implementation in the oracle's dependencies, so their reference is the definition written in numpy."""

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from statsmodels.multivariate.manova import MANOVA

REFERENCE: dict[str, str] = {"pingouin": pg.__version__, "statsmodels": statsmodels.__version__}


def mardia(x: FloatArray) -> list[float]:
    n, p = x.shape
    c = x - x.mean(0)
    s = np.cov(x, rowvar=False, ddof=0)
    d = c @ np.linalg.inv(s) @ c.T
    b1 = (d**3).sum() / n**2
    b2 = (np.diag(d) ** 2).sum() / n
    df = p * (p + 1) * (p + 2) / 6
    skew = n * b1 / 6
    z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return [float(skew), float(st.chi2.sf(skew, df)), float(z), float(2 * st.norm.sf(abs(z)))]


def cases() -> list[Case]:
    g = rng(5555)
    x = g.normal(size=(15, 3)) + np.array([0.3, 0.0, -0.2])
    y = g.normal(size=(18, 3)) * np.array([1.0, 1.4, 0.8])
    pre = g.normal(size=(12, 2))
    post = pre + g.normal(size=(12, 2)) * 0.5 + np.array([0.4, 0.1])
    out: list[Case] = []
    r = pg.multivariate_ttest(x)
    out.append(case("hotelling/one", "hotellingOne", [x, x], np.asarray([r["T2"].iloc[0], r["pval"].iloc[0]])))
    mu = np.array([0.1, 0.2, -0.1])
    r = pg.multivariate_ttest(x, Y=mu)
    out.append(case("hotelling/mu", "hotellingMu", [x, mu], np.asarray([r["T2"].iloc[0], r["pval"].iloc[0]])))
    r = pg.multivariate_ttest(post, Y=pre, paired=True)
    out.append(
        case("hotelling/paired", "hotellingPaired", [post, pre], np.asarray([r["T2"].iloc[0], r["pval"].iloc[0]]))
    )
    r = pg.multivariate_ttest(x, Y=y)
    out.append(case("hotelling/two", "hotellingTwo", [x, y], np.asarray([r["T2"].iloc[0], r["pval"].iloc[0]])))
    groups = [g.normal(size=(10, 3)), g.normal(size=(12, 3)) + np.array([0.5, 0.0, 0.3]), g.normal(size=(9, 3)) * 1.3]
    frame = pd.DataFrame(np.concatenate(groups), columns=["a", "b", "c"])
    frame["g"] = np.repeat(["g0", "g1", "g2"], [len(v) for v in groups])
    res = MANOVA.from_formula("a + b + c ~ g", data=frame).mv_test().results["g"]["stat"]
    for stat, name in [
        ("wilks", "Wilks' lambda"),
        ("pillai", "Pillai's trace"),
        ("hotellingLawley", "Hotelling-Lawley trace"),
        ("roy", "Roy's greatest root"),
    ]:
        row = res.loc[name]
        out.append(
            case(
                f"manova/{stat}",
                "manova",
                groups,
                np.asarray([row["Value"], row["F Value"], row["Num DF"], row["Den DF"], row["Pr > F"]], dtype=float),
                rtol=1e-9,
                stat=stat,
            )
        )
    bm = pg.box_m(frame, dvs=["a", "b", "c"], group="g")
    out.append(
        case(
            "boxM", "boxM", groups, np.asarray([bm["Chi2"].iloc[0], bm["pval"].iloc[0], bm["df"].iloc[0]], dtype=float)
        )
    )
    z = g.standard_t(4, size=(40, 2))
    out.append(case("mardia", "mardia", [z, z], np.asarray(mardia(z))))
    return out
