"""Intervals and model comparisons against statsmodels (Wilson and Wald proportion intervals, McNemar) and the fast
DeLong algorithm of Sun and Xu (2014) in its widely used numpy form; decision metrics from their definitions.
"""

import numpy as np
import scipy.stats as st
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def field(result: object, name: str) -> float:
    """A field of a statsmodels result bunch."""
    return float(getattr(result, name))


def midrank(x: FloatArray) -> FloatArray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = t + 1
    return out


def fast_delong(scores: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """AUCs and their covariance for the rows of `scores` (Sun and Xu's fastDeLong)."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.stack([midrank(pos[r]) for r in range(k)])
    ty = np.stack([midrank(neg[r]) for r in range(k)])
    tz = np.stack([midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def cases() -> list[Case]:
    g = rng(59)
    n = 50
    y = (g.uniform(size=n) > 0.5).astype(np.float64)
    a = y + g.normal(size=n)
    b = y + 1.5 * g.normal(size=n)
    pred_a = np.where(g.uniform(size=n) < 0.8, y, 1 - y)
    pred_b = np.where(g.uniform(size=n) < 0.65, y, 1 - y)
    acc = float((pred_a == y).mean())
    lo, hi = proportion_confint(int((pred_a == y).sum()), n, alpha=0.05, method="wilson")
    wlo, whi = proportion_confint(int((pred_a == y).sum()), n, alpha=0.1, method="normal")
    aucs, cov = fast_delong(np.stack([a, b]), y)
    var_a = float(cov[0, 0])
    z = float((aucs[0] - aucs[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    p_delong = float(2 * st.norm.sf(abs(z)))
    table = np.array(
        [
            [np.sum((pred_a == y) & (pred_b == y)), np.sum((pred_a == y) & (pred_b != y))],
            [np.sum((pred_a != y) & (pred_b == y)), np.sum((pred_a != y) & (pred_b != y))],
        ]
    )
    exact = mcnemar(table, exact=True)
    chi = mcnemar(table, exact=False, correction=True)
    chi_plain = mcnemar(table, exact=False, correction=False)
    yk = g.integers(0, 3, size=n).astype(np.float64)
    pk = np.where(g.uniform(size=n) < 0.6, yk, g.integers(0, 3, size=n)).astype(np.float64)
    costs = np.array([[0.0, 1, 4], [2, 0, 1], [8, 3, 0]])
    probs = g.dirichlet(np.ones(3), size=n)
    return [
        case("withInterval/wilson-lo", "wilsonLow", [y, pred_a], np.asarray(lo)),
        case("withInterval/wilson-hi", "wilsonHigh", [y, pred_a], np.asarray(hi)),
        case("withInterval/wald-lo", "waldLow", [y, pred_a], np.asarray(wlo), level=0.9),
        case("withInterval/wald-hi", "waldHigh", [y, pred_a], np.asarray(whi), level=0.9),
        case("accuracy", "accuracy", [y, pred_a], np.asarray(acc)),
        case("delongVariance", "delongVariance", [y, a], np.asarray(var_a)),
        case("delongTest/statistic", "delongStatistic", [y, a, b], np.asarray(z)),
        case("delongTest/pValue", "delongPValue", [y, a, b], np.asarray(p_delong)),
        case(
            "mcnemarTest/exact-statistic",
            "mcnemarStatistic",
            [y, pred_a, pred_b],
            np.asarray(field(exact, "statistic")),
            exact=True,
        ),
        case(
            "mcnemarTest/exact-p", "mcnemarPValue", [y, pred_a, pred_b], np.asarray(field(exact, "pvalue")), exact=True
        ),
        case(
            "mcnemarTest/chi2-statistic",
            "mcnemarStatistic",
            [y, pred_a, pred_b],
            np.asarray(field(chi, "statistic")),
            exact=False,
        ),
        case("mcnemarTest/chi2-p", "mcnemarPValue", [y, pred_a, pred_b], np.asarray(field(chi, "pvalue")), exact=False),
        case(
            "mcnemarTest/chi2-uncorrected-p",
            "mcnemarPValue",
            [y, pred_a, pred_b],
            np.asarray(field(chi_plain, "pvalue")),
            exact=False,
            correction=False,
        ),
        case("expectedCost", "expectedCost", [yk, pk, costs], np.asarray(costs[yk.astype(int), pk.astype(int)].mean())),
        case("bayesDecision", "bayesDecision", [probs, costs], (probs @ costs).argmin(axis=1).astype(np.float64)),
    ]
