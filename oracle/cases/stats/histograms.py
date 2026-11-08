"""Histograms and bin rules against numpy.histogram, numpy.histogram_bin_edges, numpy.histogram2d and
scipy.stats.binned_statistic. Knuth's rule has no numpy or scipy implementation, so its reference is the exhaustive
maximiser of Knuth's log posterior written from the paper."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.special import gammaln

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def knuth_edges(x: FloatArray) -> FloatArray:
    n = x.size
    best_m, best = 1, -np.inf
    for m in range(1, min(n, 1000) + 1):
        counts, _ = np.histogram(x, bins=m)
        value = n * np.log(m) + gammaln(m / 2) - m * gammaln(0.5) - gammaln(n + m / 2) + np.sum(gammaln(counts + 0.5))
        if value > best:
            best_m, best = m, value
    return np.histogram_bin_edges(x, bins=best_m)


def cases() -> list[Case]:
    g = rng(12)
    x = np.concatenate([g.normal(size=60), g.normal(3.0, 0.5, size=40)])
    w = g.uniform(0.1, 2.0, size=x.size)
    y = 0.5 * x + g.normal(size=x.size)
    out: list[Case] = []
    for rule in ["auto", "fd", "scott", "sturges", "doane", "rice", "sqrt"]:
        out.append(case(f"edges/{rule}", "edges", [x], np.histogram_bin_edges(x, bins=rule), bins=rule))
        out.append(
            case(
                f"edges/range/{rule}",
                "edges",
                [x],
                np.histogram_bin_edges(x, bins=rule, range=(-1.0, 2.0)),
                bins=rule,
                lo=-1.0,
                hi=2.0,
            )
        )
    out.append(case("edges/knuth", "edges", [x], knuth_edges(x), bins="knuth"))
    counts, _ = np.histogram(x, bins=12)
    dens, _ = np.histogram(x, bins=12, density=True)
    wc, _ = np.histogram(x, bins=7, weights=w, range=(-2.0, 3.0))
    explicit = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 4.5])
    ec, _ = np.histogram(x, bins=explicit)
    h2, _, _ = np.histogram2d(x, y, bins=(5, 4))
    h2d, _, _ = np.histogram2d(x, y, bins=(5, 4), density=True, weights=w)
    out += [
        case("histogram/counts", "histogram", [x], counts.astype(float), bins=12),
        case("histogram/density", "histogram", [x], dens, bins=12, density=True),
        case("histogram/weighted", "histogram", [x, w], wc, bins=7, lo=-2.0, hi=3.0),
        case("histogram/edges", "histogramEdges", [x, explicit], ec.astype(float)),
        case("histogram2d", "histogram2d", [x, y], h2, bx=5, by=4),
        case("histogram2d/density", "histogram2d", [x, y, w], h2d, bx=5, by=4, density=True),
    ]
    stats: list[Literal["mean", "median", "count", "sum", "std", "min", "max"]] = [
        "mean",
        "median",
        "count",
        "sum",
        "std",
        "min",
        "max",
    ]
    for stat in stats:
        r = st.binned_statistic(x, y, statistic=stat, bins=6)
        out.append(case(f"binnedStatistic/{stat}", "binnedStatistic", [x, y], r.statistic, statistic=stat, bins=6))
    r = st.binned_statistic(x, y, bins=5, range=(-1.0, 2.5))
    out.append(
        case(
            "binnedStatistic/binNumber",
            "binNumber",
            [x, y],
            r.binnumber.astype(float),
            bins=5,
            lo=-1.0,
            hi=2.5,
        )
    )
    return out
