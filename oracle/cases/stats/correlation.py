"""Correlations and covariances against scipy.stats, numpy and pingouin (partial and distance correlation)."""

from typing import Literal

import numpy as np
import pandas as pd
import pingouin as pg
import scipy
import scipy.stats as st
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "pingouin": pg.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]
VARIANTS: list[Literal["b", "c"]] = ["b", "c"]


def cases() -> list[Case]:
    g = rng(1234)
    n = 30
    x = g.normal(size=n)
    y = 0.5 * x + g.normal(size=n)
    z = g.normal(size=(n, 2))
    y2 = y + z @ np.array([0.7, -0.4])
    xt = np.round(x * 2) / 2
    yt = np.round(y * 2) / 2
    small = g.permutation(np.arange(12.0))
    small2 = small + g.normal(size=12) * 3
    out: list[Case] = []
    for alt in ALTS:
        r = st.pearsonr(x, y, alternative=alt)
        ci = r.confidence_interval(0.9)
        out.append(
            case(
                f"pearson/{alt}",
                "pearson",
                [x, y],
                np.asarray([r.statistic, r.pvalue, ci.low, ci.high]),
                rtol=1e-10,
                alternative=alt,
            )
        )
        r = st.spearmanr(xt, y, alternative=alt)
        out.append(
            case(
                f"spearman/{alt}", "spearman", [xt, y], np.asarray([r.statistic, r.pvalue]), rtol=1e-10, alternative=alt
            )
        )
        for name, (a, b) in {"exact": (small, small2), "ties": (xt, yt), "large": (x, y)}.items():
            for variant in VARIANTS:
                r = st.kendalltau(a, b, alternative=alt, variant=variant)
                out.append(
                    case(
                        f"kendallTau/{name}/{variant}/{alt}",
                        "kendallTau",
                        [a, b],
                        np.asarray([r.statistic, r.pvalue]),
                        rtol=1e-10,
                        alternative=alt,
                        variant=variant,
                    )
                )
        frame = pd.DataFrame({"x": x, "y": y2, "z1": z[:, 0], "z2": z[:, 1]})
        for method in ["pearson", "spearman"]:
            pc = pg.partial_corr(data=frame, x="x", y="y", covar=["z1", "z2"], method=method, alternative=alt)
            out.append(
                case(
                    f"partialCorrelation/{method}/{alt}",
                    "partialCorrelation",
                    [x, y2, z],
                    np.asarray([np.asarray(pc["r"])[0], np.asarray(pc["p_val"])[0]]),
                    rtol=1e-9,
                    method=method,
                    alternative=alt,
                )
            )
    pc = pg.partial_corr(
        data=pd.DataFrame({"x": x, "y": y2, "z1": z[:, 0], "z2": z[:, 1]}), x="x", y="y", covar=["z1", "z2"]
    )
    ci = np.asarray(pc["CI95"])[0]
    # pingouin rounds the interval it reports to 2 decimals.
    out.append(
        case("partialCorrelation/ci", "partialCorrelationCi", [x, y2, z], np.asarray(ci), rtol=0, atol=5.0001e-3)
    )
    b = (g.uniform(size=n) < 0.4).astype(float)
    r = st.pointbiserialr(b, x)
    out.append(case("pointBiserial", "pointBiserial", [b, x], np.asarray([r.statistic, r.pvalue])))
    stack = np.stack([x, y, x - y])
    fw = g.integers(1, 4, size=n)
    aw = g.uniform(0.5, 2.0, size=n)
    out += [
        case("cov", "cov", [stack, stack], np.cov(stack)),
        case("cov/ddof0", "cov", [stack, stack], np.cov(stack, ddof=0), ddof=0),
        case("cov/columns", "covColumns", [stack.T, stack.T], np.cov(stack.T, rowvar=False)),
        case("cov/xy", "covXY", [x, y], np.cov(x, y)),
        case("cov/weights", "covWeights", [stack, fw.astype(float), aw], np.cov(stack, fweights=fw, aweights=aw)),
        case("cov/aweights", "covAweights", [stack, aw], np.cov(stack, aweights=aw)),
        case("corrcoef", "corrcoef", [stack, stack], np.corrcoef(stack)),
    ]
    xy = np.stack([x, np.abs(x)], axis=1)
    out.append(case("distanceCorrelation", "distanceCorrelation", [x, y], np.asarray(pg.distance_corr(x, y, n_boot=1))))
    out.append(
        case(
            "distanceCorrelation/multivariate",
            "distanceCorrelation",
            [xy, z],
            np.asarray(pg.distance_corr(xy, z, n_boot=1)),
        )
    )
    return out
