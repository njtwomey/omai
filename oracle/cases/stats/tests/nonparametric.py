"""Rank and sign tests against scipy.stats and statsmodels (sign test). The Nemenyi post-hoc tests have no scipy or
statsmodels implementation (scikit-posthocs is not an oracle dependency), so their references are scikit-posthocs'
formulas written with scipy.stats.studentized_range."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from statsmodels.stats.descriptivestats import sign_test
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]
ZEROS: list[Literal["wilcox", "pratt", "zsplit"]] = ["wilcox", "pratt", "zsplit"]


def nemenyi_friedman(data: FloatArray) -> tuple[FloatArray, FloatArray]:
    n, k = data.shape
    ranks = np.apply_along_axis(st.rankdata, 1, data)
    mean = ranks.mean(0)
    q = np.abs(mean[:, None] - mean[None, :]) / np.sqrt(k * (k + 1) / (6 * n))
    p = st.studentized_range.sf(q * np.sqrt(2), k, np.inf)
    np.fill_diagonal(p, 1.0)
    return q, p


def nemenyi_kruskal(groups: list[FloatArray]) -> tuple[FloatArray, FloatArray]:
    allx = np.concatenate(groups)
    n = allx.size
    ranks = st.rankdata(allx)
    sizes = np.array([g.size for g in groups])
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    mean = np.array([ranks[bounds[i] : bounds[i + 1]].mean() for i in range(len(groups))])
    c = st.tiecorrect(ranks)  # pyright: ignore[reportArgumentType]
    scale = np.sqrt(n * (n + 1) / 12 * c * (1 / sizes[:, None] + 1 / sizes[None, :]))
    q = np.abs(mean[:, None] - mean[None, :]) / scale
    p = st.studentized_range.sf(q * np.sqrt(2), len(groups), np.inf)
    np.fill_diagonal(p, 1.0)
    return q, p


def cases() -> list[Case]:
    g = rng(5150)
    small_x = g.normal(0, 1, 6)
    small_y = g.normal(1, 1, 7)
    x = g.normal(0, 1, 25)
    y = g.normal(0.5, 1.3, 22)
    tx = np.round(x)
    ty = np.round(y)
    d = g.normal(0.4, 1, 18)
    dt = np.round(d * 2) / 2
    dsmall = np.array([1.5, -0.5, 2.0, 0.0, 1.5, 3.0, -1.0, 0.5, 2.5, 1.0])
    out: list[Case] = []
    for alt in ALTS:
        for name, (a, b) in {"exact": (small_x, small_y), "asymptotic": (x, y), "ties": (tx, ty)}.items():
            r = st.mannwhitneyu(a, b, alternative=alt)
            out.append(
                case(
                    f"mannWhitneyU/{name}/{alt}",
                    "mannWhitneyU",
                    [a, b],
                    np.asarray([r.statistic, r.pvalue]),
                    alternative=alt,
                )
            )
        r = st.mannwhitneyu(x, y, alternative=alt, use_continuity=False)
        out.append(
            case(
                f"mannWhitneyU/nocontinuity/{alt}",
                "mannWhitneyU",
                [x, y],
                np.asarray([r.statistic, r.pvalue]),
                alternative=alt,
                useContinuity=False,
            )
        )
        for name, data in {"exact": d, "ties": dt, "small": dsmall}.items():
            for zero in ZEROS:
                r = st.wilcoxon(data, alternative=alt, zero_method=zero)
                out.append(
                    case(
                        f"wilcoxon/{name}/{zero}/{alt}",
                        "wilcoxon",
                        [data, data],
                        np.asarray([r.statistic, r.pvalue]),
                        alternative=alt,
                        zeroMethod=zero,
                    )
                )
        r = st.wilcoxon(dt, alternative=alt, method="asymptotic", correction=True)
        out.append(
            case(
                f"wilcoxon/asymptotic/corrected/{alt}",
                "wilcoxon",
                [dt, dt],
                np.asarray([r.statistic, r.pvalue]),
                alternative=alt,
                method="asymptotic",
                correction=True,
            )
        )
        r = st.brunnermunzel(x, y, alternative=alt)
        out.append(
            case(f"brunnerMunzel/{alt}", "brunnerMunzel", [x, y], np.asarray([r.statistic, r.pvalue]), alternative=alt)
        )
        r = st.brunnermunzel(x, y, alternative=alt, distribution="normal")
        out.append(
            case(
                f"brunnerMunzel/normal/{alt}",
                "brunnerMunzel",
                [x, y],
                np.asarray([r.statistic, r.pvalue]),
                alternative=alt,
                distribution="normal",
            )
        )
    a, b, c = g.normal(0, 1, 9), np.round(g.normal(0.8, 1, 11), 1), g.normal(0.2, 1, 8)
    r = st.kruskal(a, b, c)
    out.append(case("kruskal", "kruskal", [a, b, c], np.asarray([r.statistic, r.pvalue])))
    blocks = np.round(g.normal(size=(12, 4)) + np.array([0.0, 0.4, 0.8, 0.2]), 1)
    r = st.friedmanchisquare(*blocks.T)
    out.append(case("friedman", "friedman", [blocks, blocks], np.asarray([r.statistic, r.pvalue])))
    q, p = nemenyi_friedman(blocks)
    out.append(case("nemenyiFriedman/q", "nemenyiFriedmanQ", [blocks, blocks], q))
    out.append(case("nemenyiFriedman/p", "nemenyiFriedmanP", [blocks, blocks], p, rtol=1e-7))
    q, p = nemenyi_kruskal([a, b, c])
    out.append(case("nemenyi/q", "nemenyiQ", [a, b, c], q))
    out.append(case("nemenyi/p", "nemenyiP", [a, b, c], p, rtol=1e-7))
    m, p = sign_test(d, 0.2)  # pyright: ignore[reportArgumentType]
    out.append(case("signTest", "signTest", [d, d], np.asarray([m, p]), mu0=0.2))
    return out
