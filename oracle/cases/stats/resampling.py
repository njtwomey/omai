"""Exact permutation tests against scipy.stats.permutation_test (small samples, where scipy enumerates every
arrangement), and the jackknife from its definition (neither numpy nor scipy has one)."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

Alt = Literal["two-sided", "less", "greater"]
ALTS: list[Alt] = ["two-sided", "less", "greater"]


def mean_difference(x: FloatArray, y: FloatArray, axis: int) -> FloatArray:
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def mean_stat(x: FloatArray, axis: int) -> FloatArray:
    return np.mean(x, axis=axis)


def correlation(x: FloatArray, y: FloatArray, axis: int) -> FloatArray:
    return np.asarray(st.pearsonr(x, y, axis=axis).statistic)


def cases() -> list[Case]:
    g = rng(17)
    a = g.normal(0, 1, 4)
    b = g.normal(0.8, 1, 5)
    c = g.normal(0.3, 1, 3)
    d = g.normal(0.5, 1, 7)
    px = g.normal(0, 1, 6)
    py = px + g.normal(0.4, 0.5, 6)
    out: list[Case] = []
    for alt in ALTS:
        r = st.permutation_test((a, b), mean_difference, alternative=alt, vectorized=True)
        out.append(
            case(f"independent/{alt}", "independent", [a, b], np.asarray([r.statistic, r.pvalue]), alternative=alt)
        )
        r = st.permutation_test((d,), mean_stat, permutation_type="samples", alternative=alt, vectorized=True)
        out.append(
            case(f"samples/one/{alt}", "samplesOne", [d, d], np.asarray([r.statistic, r.pvalue]), alternative=alt)
        )
        r = st.permutation_test((px, py), mean_difference, permutation_type="samples", alternative=alt, vectorized=True)
        out.append(
            case(f"samples/two/{alt}", "samplesTwo", [px, py], np.asarray([r.statistic, r.pvalue]), alternative=alt)
        )
        r = st.permutation_test(
            (px[:4], py[:4]), correlation, permutation_type="pairings", alternative=alt, vectorized=True
        )
        out.append(
            case(f"pairings/{alt}", "pairings", [px[:4], py[:4]], np.asarray([r.statistic, r.pvalue]), alternative=alt)
        )

    def f_stat(*s: FloatArray) -> float:
        return float(st.f_oneway(*s).statistic)

    r = st.permutation_test((a[:3], b[:3], c), f_stat, alternative="greater")
    out.append(case("independent/three", "independentThree", [a[:3], b[:3], c], np.asarray([r.statistic, r.pvalue])))
    x = g.gamma(2.0, 1.0, 15)
    n = x.size
    loo = np.array([np.var(np.delete(x, i), ddof=1) for i in range(n)])
    theta = np.var(x, ddof=1)
    bias = (n - 1) * (loo.mean() - theta)
    se = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
    out.append(case("jackknife", "jackknife", [x, x], np.asarray([theta - bias, se, bias])))
    return out
