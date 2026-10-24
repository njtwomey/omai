"""Goodness-of-fit tests against scipy.stats and statsmodels (Lilliefors)."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from statsmodels.stats.diagnostic import lilliefors
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]
Dist = Literal["norm", "expon", "logistic", "gumbel_r", "gumbel_l"]


def cases() -> list[Case]:
    g = rng(808)
    out: list[Case] = []
    for n in [8, 30, 120, 400]:
        x = g.normal(0.15, 1.1, size=n)
        for alt in ALTS:
            r = st.ks_1samp(x, st.norm.cdf, alternative=alt)
            out.append(
                case(
                    f"kstest/{n}/{alt}",
                    "kstest",
                    [x, x],
                    np.asarray([r.statistic, r.pvalue]),
                    rtol=1e-9,
                    alternative=alt,
                )
            )
        r = st.ks_1samp(x, st.norm.cdf, method="asymp")
        out.append(
            case(f"kstest/{n}/asymp", "kstest", [x, x], np.asarray([r.statistic, r.pvalue]), rtol=1e-9, method="asymp")
        )
    samples: dict[Dist, np.ndarray] = {
        "norm": g.normal(2, 1, 40),
        "expon": g.exponential(2.0, 40),
        "logistic": g.logistic(1.0, 0.7, 40),
        "gumbel_r": g.gumbel(0.5, 1.2, 40),
        "gumbel_l": -g.gumbel(0.5, 1.2, 40),
    }
    for dist, x in samples.items():
        for label, data in [("fit", x), ("misfit", x**2)]:
            # scipy accepts method="interpolate"; its stubs spell it "interpolated".
            r = st.anderson(data, dist=dist, method="interpolate")  # pyright: ignore[reportCallIssue, reportArgumentType]
            # scipy fits the logistic by fsolve with xtol 1e-5; the statistic inherits that accuracy.
            tol = 1e-4 if dist == "logistic" else 1e-9
            out.append(
                case(
                    f"andersonDarling/{dist}/{label}",
                    "andersonDarling",
                    [data, data],
                    np.asarray([r.statistic, r.pvalue]),
                    rtol=tol,
                    dist=dist,
                )
            )
    for n in [3, 7, 11, 25, 120]:
        x = g.normal(size=n) if n != 25 else g.exponential(size=n)
        r = st.shapiro(x)
        out.append(case(f"shapiro/{n}", "shapiro", [x, x], np.asarray([r.statistic, r.pvalue]), rtol=1e-6))
    x = g.standard_t(5, size=60)
    r = st.jarque_bera(x)
    out.append(case("jarqueBera", "jarqueBera", [x, x], np.asarray([r.statistic, r.pvalue])))
    counts = np.array([16.0, 18.0, 16.0, 14.0, 12.0, 12.0])
    expected = np.array([14.0, 14.0, 16.0, 16.0, 14.0, 14.0])
    r = st.chisquare(counts)
    out.append(case("chisquare/uniform", "chisquare", [counts, counts], np.asarray([r.statistic, r.pvalue])))
    r = st.chisquare(counts, expected, ddof=1)
    out.append(
        case("chisquare/expected", "chisquareExpected", [counts, expected], np.asarray([r.statistic, r.pvalue]), ddof=1)
    )
    lams: list[Literal["log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"]] = [
        "log-likelihood",
        "freeman-tukey",
        "mod-log-likelihood",
        "neyman",
        "cressie-read",
    ]
    for lam in lams:
        r = st.power_divergence(counts, expected, lambda_=lam)
        out.append(
            case(
                f"powerDivergence/{lam}",
                "powerDivergence",
                [counts, expected],
                np.asarray([r.statistic, r.pvalue]),
                lam=lam,
            )
        )
    skewed = g.exponential(size=40) + 1.0
    d, p = lilliefors(skewed, dist="norm", pvalmethod="approx")
    out.append(case("lilliefors/norm", "lilliefors", [skewed, skewed], np.asarray([d, p]), rtol=1e-10))
    d, _ = lilliefors(skewed, dist="exp")
    out.append(case("lilliefors/exp/statistic", "lillieforsStatistic", [skewed, skewed], np.asarray(d), dist="exp"))
    u = g.uniform(size=20) ** 1.3
    r = st.cramervonmises(u, "uniform")
    out.append(case("cramervonmises", "cramervonmises", [u, u], np.asarray([r.statistic, r.pvalue]), rtol=1e-8))
    return out
