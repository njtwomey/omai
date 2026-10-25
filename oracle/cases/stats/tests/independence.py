"""Independence and conditional independence tests against pingouin (distance correlation), sklearn (discrete mutual
information) and causal-learn (Fisher's z, G-square, chi-square and KCI). Hoeffding's D and the binned mutual
information follow their definitions (Hollander and Wolfe; the plug-in estimator) in numpy."""

from importlib.metadata import version

import numpy as np
import pingouin as pg
import sklearn
from causallearn.utils.cit import CIT
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.metrics import mutual_info_score

REFERENCE: dict[str, str] = {
    "pingouin": pg.__version__,
    "sklearn": sklearn.__version__,
    "causal-learn": version("causal-learn"),
}


def midranks(v: FloatArray) -> FloatArray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size)
    s = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def hoeffding(x: FloatArray, y: FloatArray) -> float:
    n = x.size
    R = midranks(x)
    S = midranks(y)
    Q = np.empty(n)
    for i in range(n):
        lx = x < x[i]
        ex = x == x[i]
        ly = y < y[i]
        ey = y == y[i]
        q = 1 + np.sum(lx & ly) + 0.5 * np.sum(ex & ly) + 0.5 * np.sum(lx & ey) + 0.25 * (np.sum(ex & ey) - 1)
        Q[i] = q
    D1 = np.sum((Q - 1) * (Q - 2))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * (Q - 1))
    return float(30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3) / (n * (n - 1) * (n - 2) * (n - 3) * (n - 4)))


def binned_mi(x: FloatArray, y: FloatArray, bins: int) -> float:
    c, _, _ = np.histogram2d(x, y, bins=bins)
    n = c.sum()
    pr = c.sum(1, keepdims=True)
    pc = c.sum(0, keepdims=True)
    nz = c > 0
    return float(np.sum(c[nz] / n * np.log(c[nz] * n / (pr @ pc)[nz])))


def cases() -> list[Case]:
    g = rng(909)
    x = g.normal(size=25)
    y = np.sin(2 * x) + 0.3 * g.normal(size=25)
    xt = np.round(x * 2) / 2
    yt = np.round(y * 2) / 2
    out: list[Case] = [
        case("distanceCorrelation", "distanceCovariance", [x, y], np.asarray(pg.distance_corr(x, y, n_boot=1))),
        case("hoeffding", "hoeffding", [x, y], np.asarray(hoeffding(x, y))),
        case("hoeffding/ties", "hoeffding", [xt, yt], np.asarray(hoeffding(xt, yt))),
        case("mutualInformation/discrete", "miDiscrete", [xt, yt], np.asarray(mutual_info_score(xt, yt))),
        case("mutualInformation/binned", "miBinned", [x, y], np.asarray(binned_mi(x, y, 5)), bins=5),
    ]
    n = 120
    z1 = g.normal(size=n)
    z2 = g.normal(size=n)
    a = z1 + 0.5 * g.normal(size=n)
    b = z1 - z2 + 0.5 * g.normal(size=n)
    c = a + z2 + 0.5 * g.normal(size=n)
    data = np.stack([a, b, c, z1, z2], axis=1)
    fz = CIT(data, "fisherz")  # pyright: ignore[reportCallIssue]
    for xi, yi, zs in [(0, 1, []), (0, 1, [3]), (0, 2, [3, 4]), (1, 2, [0, 3, 4])]:
        label = f"{xi}-{yi}-{'-'.join(map(str, zs)) or 'none'}"
        out.append(
            case(
                f"fisherz/{label}",
                "fisherz",
                [data, data],
                np.asarray(fz(xi, yi, zs)),  # pyright: ignore[reportCallIssue]
                x=xi,
                y=yi,
                z=np.asarray(zs, dtype=np.int64),
            )
        )
    disc = np.stack(
        [
            (a > 0).astype(float),
            (b > 0.5).astype(float) + (b > -0.5),
            (c > 0).astype(float),
            (z1 > 0).astype(float) + (z1 > 1),
            (z2 > 0).astype(float),
        ],
        axis=1,
    )
    for method in ["gsq", "chisq"]:
        test = CIT(disc, method)  # pyright: ignore[reportCallIssue]
        for xi, yi, zs in [(0, 1, []), (0, 1, [3]), (0, 2, [3, 4])]:
            label = f"{xi}-{yi}-{'-'.join(map(str, zs)) or 'none'}"
            out.append(
                case(
                    f"{method}/{label}",
                    "gTestConditional",
                    [disc, disc],
                    np.asarray(test(xi, yi, zs)),  # pyright: ignore[reportCallIssue]
                    rtol=1e-10,
                    x=xi,
                    y=yi,
                    z=np.asarray(zs, dtype=np.int64),
                    statistic="g" if method == "gsq" else "chi2",
                )
            )
    small = data[:60]
    kci = CIT(small, "kci")  # pyright: ignore[reportCallIssue]
    for xi, yi, zs in [(0, 1, [3]), (0, 2, [3, 4])]:
        label = f"{xi}-{yi}-{'-'.join(map(str, zs))}"
        # The eigen-decompositions of the residual kernels differ between LAPACK builds in the last digits.
        out.append(
            case(
                f"kci/{label}",
                "kci",
                [small, small],
                np.asarray(kci(xi, yi, zs)),  # pyright: ignore[reportCallIssue]
                rtol=1e-6,
                x=xi,
                y=yi,
                z=np.asarray(zs, dtype=np.int64),
            )
        )
    return out
