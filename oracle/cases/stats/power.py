"""Power and sample size against statsmodels.stats.power."""

from typing import Literal

import numpy as np
import statsmodels
from mdml_oracle.fixture import Case, case
from statsmodels.stats.power import FTestAnovaPower, GofChisquarePower, NormalIndPower, TTestIndPower, TTestPower

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}

Alt = Literal["two-sided", "larger", "smaller"]
ALTS: dict[str, Alt] = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}


def cases() -> list[Case]:
    out: list[Case] = []
    for ours, theirs in ALTS.items():
        for d, n in [(0.5, 30), (0.2, 100), (-0.4, 25)]:
            p = TTestPower().power(effect_size=d, nobs=n, alpha=0.05, alternative=theirs)
            out.append(
                case(
                    f"ttest/one/{ours}/{d}/{n}",
                    "ttest",
                    [],
                    np.asarray(p),
                    rtol=1e-9,
                    d=d,
                    n=n,
                    kind="one-sample",
                    alternative=ours,
                )
            )
            p = TTestIndPower().power(effect_size=d, nobs1=n, alpha=0.05, ratio=1.5, alternative=theirs)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"ttest/two/{ours}/{d}/{n}",
                    "ttest",
                    [],
                    np.asarray(p),
                    rtol=1e-9,
                    d=d,
                    n=n,
                    kind="two-sample",
                    ratio=1.5,
                    alternative=ours,
                )
            )
            p = NormalIndPower().power(effect_size=d, nobs1=n, alpha=0.05, ratio=1.5, alternative=theirs)  # pyright: ignore[reportArgumentType]
            out.append(
                case(
                    f"proportions/{ours}/{d}/{n}",
                    "proportions",
                    [],
                    np.asarray(p),
                    d=d,
                    n=n,
                    ratio=1.5,
                    alternative=ours,
                )
            )
    n = TTestIndPower().solve_power(effect_size=0.5, power=0.8, alpha=0.05)
    out.append(case("sampleSize/ttest/two", "sampleSizeTtest", [], np.asarray(n), rtol=1e-7, kind="two-sample"))
    n = TTestPower().solve_power(effect_size=0.3, power=0.9, alpha=0.05)
    out.append(case("sampleSize/ttest/one", "sampleSizeTtestOne", [], np.asarray(n), rtol=1e-7))
    n = NormalIndPower().solve_power(effect_size=0.3, power=0.8, alpha=0.05)
    out.append(case("sampleSize/proportions", "sampleSizeProportions", [], np.asarray(n), rtol=1e-7))
    for f, n, k in [(0.25, 120, 3), (0.1, 400, 4)]:
        p = FTestAnovaPower().power(effect_size=f, nobs=n, alpha=0.05, k_groups=k)
        out.append(case(f"anova/{f}/{n}/{k}", "anova", [], np.asarray(p), rtol=1e-8, f=f, n=n, k=k))
    n = FTestAnovaPower().solve_power(effect_size=0.25, power=0.8, alpha=0.05, k_groups=3)
    out.append(case("sampleSize/anova", "sampleSizeAnova", [], np.asarray(n), rtol=1e-6))
    for w, n, bins in [(0.3, 100, 4), (0.1, 500, 6)]:
        p = GofChisquarePower().power(effect_size=w, nobs=n, alpha=0.05, n_bins=bins)
        out.append(case(f"chisquare/{w}/{n}/{bins}", "chisquare", [], np.asarray(p), rtol=1e-8, w=w, n=n, bins=bins))
    n = GofChisquarePower().solve_power(effect_size=0.3, power=0.8, alpha=0.05, n_bins=4)
    out.append(case("sampleSize/chisquare", "sampleSizeChisquare", [], np.asarray(n), rtol=1e-6))
    return out
