"""Effect sizes against pingouin (Cohen's d, Hedges' g and their intervals), statsmodels (Cohen's h) and scipy
(Cramer's V). Glass's delta, omega squared, Cliff's delta and the noncentrality intervals have no library function in
the oracle's dependencies, so their references are the published formulas written with scipy's noncentral chi-square
and F distributions and brentq."""

from collections.abc import Callable

import numpy as np
import pingouin as pg
import scipy
import scipy.stats as st
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.optimize import brentq
from scipy.stats.contingency import association
from statsmodels.stats.proportion import proportion_effectsize

REFERENCE: dict[str, str] = {
    "pingouin": pg.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
}


def bounds(cdf: Callable[[float], float], level: float) -> tuple[float, float]:
    hi, lo = (1 + level) / 2, (1 - level) / 2

    def solve(target: float) -> float:
        if cdf(0.0) <= target:
            return 0.0
        b = 1.0
        while cdf(b) > target:
            b *= 2
        return float(brentq(lambda lam: cdf(lam) - target, 0.0, b, xtol=1e-14, rtol=1e-14))

    return solve(hi), solve(lo)


def one_way(groups: list[FloatArray]) -> tuple[float, float, int, int]:
    allx = np.concatenate(groups)
    grand = allx.mean()
    ssb = sum(gr.size * (gr.mean() - grand) ** 2 for gr in groups)
    ssw = sum(((gr - gr.mean()) ** 2).sum() for gr in groups)
    return float(ssb), float(ssw), len(groups), allx.size


def cliffs(x: FloatArray, y: FloatArray, level: float) -> list[float]:
    s = np.sign(x[:, None] - y[None, :])
    d = s.mean()
    nx, ny = s.shape
    di = s.mean(1)
    dj = s.mean(0)
    si = ((di - d) ** 2).sum() / (nx - 1)
    sj = ((dj - d) ** 2).sum() / (ny - 1)
    sij = ((s - d) ** 2).sum() / ((nx - 1) * (ny - 1))
    var = ((ny - 1) * si + (nx - 1) * sj + sij) / (nx * ny)
    z = st.norm.ppf(0.5 + level / 2)
    root = z * np.sqrt(var) * np.sqrt((1 - d**2) ** 2 + z**2 * var)
    den = 1 - d**2 + z**2 * var
    return [float(d), float((d - d**3 - root) / den), float((d - d**3 + root) / den)]


def cases() -> list[Case]:
    g = rng(2718)
    x = g.normal(1.0, 1.2, 15)
    y = g.normal(0.2, 0.9, 12)
    pre = g.normal(5, 1, 10)
    post = pre + g.normal(0.5, 0.8, 10)
    out: list[Case] = []
    for eftype in ["cohen", "hedges"]:
        for label, a, b, paired in [("independent", x, y, False), ("paired", post, pre, True)]:
            es = pg.compute_effsize(a, b, paired=paired, eftype=eftype)
            ci = pg.compute_esci(es, a.size, b.size, paired=paired, eftype=eftype, confidence=0.9, decimals=12)
            out.append(
                case(f"{eftype}/{label}", eftype, [a, b], np.asarray([es, ci[0], ci[1]]), rtol=1e-10, paired=paired)
            )
        es = pg.compute_effsize(x, np.array([0.3]), eftype=eftype)
        ci = pg.compute_esci(es, x.size, 1, eftype=eftype, confidence=0.9, decimals=12)
        out.append(
            case(f"{eftype}/one-sample", f"{eftype}One", [x, x], np.asarray([es, ci[0], ci[1]]), rtol=1e-10, mu=0.3)
        )
    delta = (x.mean() - y.mean()) / y.std(ddof=1)
    se = np.sqrt((x.size + y.size) / (x.size * y.size) + delta**2 / (2 * (y.size - 1)))
    z = st.norm.ppf(0.95)
    out.append(case("glassDelta", "glassDelta", [x, y], np.asarray([delta, delta - z * se, delta + z * se])))
    h = proportion_effectsize(0.62, 0.41)
    se = np.sqrt(1 / 50 + 1 / 60)
    out.append(case("cohensH", "cohensH", [], np.asarray([h, h - z * se, h + z * se]), p1=0.62, p2=0.41, n1=50, n2=60))
    table = np.array([[20.0, 15.0, 8.0], [9.0, 18.0, 22.0]])
    chi2 = st.chi2_contingency(table, correction=False).statistic
    n = table.sum()
    df = 2
    lo, hi = bounds(lambda lam: float(st.chi2.cdf(chi2, df) if lam == 0 else st.ncx2.cdf(chi2, df, lam)), 0.9)
    v = association(table.astype(int), method="cramer")
    out.append(
        case(
            "cramersV",
            "cramersV",
            [table, table],
            np.asarray([v, np.sqrt(lo / (n * 1)), np.sqrt(hi / (n * 1))]),
            rtol=1e-8,
        )
    )
    r, c = table.shape
    phi2 = max(0.0, chi2 / n - df / (n - 1))
    rr, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    out.append(
        case("cramersV/corrected", "cramersVCorrected", [table, table], np.asarray(np.sqrt(phi2 / min(rr - 1, cc - 1))))
    )
    groups = [g.normal(0, 1, 9), g.normal(0.9, 1, 11), g.normal(0.4, 1, 10)]
    ssb, ssw, k, nn = one_way(groups)
    F = ssb / (k - 1) / (ssw / (nn - k))
    lo, hi = bounds(
        lambda lam: float(st.f.cdf(F, k - 1, nn - k) if lam == 0 else st.ncf.cdf(F, k - 1, nn - k, lam)), 0.9
    )
    N = nn
    eta = (lo / (lo + N), hi / (hi + N))
    msw = ssw / (nn - k)
    out.append(case("etaSquared", "etaSquared", groups, np.asarray([ssb / (ssb + ssw), *eta]), rtol=1e-8))
    out.append(
        case(
            "omegaSquared",
            "omegaSquared",
            groups,
            np.asarray([(ssb - (k - 1) * msw) / (ssb + ssw + msw), *eta]),
            rtol=1e-8,
        )
    )
    out.append(case("cliffsDelta", "cliffsDelta", [x, y], np.asarray(cliffs(x, y, 0.9))))
    return out
