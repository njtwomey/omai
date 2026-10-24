"""Quantiles (the nine Hyndman-Fan methods), IQR, percentile ranks, the ECDF with Greenwood intervals and the MAD,
against numpy.quantile and scipy.stats."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

METHODS: list[
    Literal[
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
    ]
] = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
]


def cases() -> list[Case]:
    g = rng(2024)
    x = g.normal(size=17)
    ties = np.round(g.normal(size=20), 0)
    q = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 0.99, 1.0])
    out: list[Case] = []
    for method in METHODS:
        out.append(case(f"quantile/{method}", "quantile", [x, q], np.quantile(x, q, method=method), method=method))
        out.append(
            case(f"quantile/ties/{method}", "quantile", [ties, q], np.quantile(ties, q, method=method), method=method)
        )
    out += [
        case("median", "median", [x], np.asarray(np.median(x))),
        case("iqr", "iqr", [x], np.asarray(st.iqr(x))),
        case("iqr/normal", "iqr", [x], np.asarray(st.iqr(x, scale="normal")), scale="normal"),
        case("iqr/rng", "iqr", [x], np.asarray(st.iqr(x, rng=(10, 90))), lo=10, hi=90),
    ]
    kinds: list[Literal["rank", "weak", "strict", "mean"]] = ["rank", "weak", "strict", "mean"]
    for kind in kinds:
        scores = [-0.5, 0.0, float(ties[3]), 2.0]
        out.append(
            case(
                f"percentileOfScore/{kind}",
                "percentileOfScore",
                [ties, np.asarray(scores)],
                np.asarray([st.percentileofscore(ties, s, kind=kind) for s in scores]),
                kind=kind,
            )
        )
    e = st.ecdf(ties)
    lin = e.cdf.confidence_interval(0.9)
    loglog = e.cdf.confidence_interval(0.9, method="log-log")
    out += [
        case("ecdf/quantiles", "ecdfQuantiles", [ties], np.asarray(e.cdf.quantiles)),
        case("ecdf/probabilities", "ecdfProbabilities", [ties], np.asarray(e.cdf.probabilities)),
        case("ecdf/evaluate", "ecdfEvaluate", [ties, q * 4 - 2], np.asarray(e.cdf.evaluate(q * 4 - 2))),
        case("ecdf/linear/low", "ecdfLow", [ties], np.asarray(lin.low.probabilities), method="linear"),
        case("ecdf/linear/high", "ecdfHigh", [ties], np.asarray(lin.high.probabilities), method="linear"),
        case("ecdf/loglog/low", "ecdfLow", [ties], np.asarray(loglog.low.probabilities), method="log-log"),
        case("ecdf/loglog/high", "ecdfHigh", [ties], np.asarray(loglog.high.probabilities), method="log-log"),
        case("mad", "medianAbsoluteDeviation", [x], np.asarray(st.median_abs_deviation(x))),
        case(
            "mad/normal",
            "medianAbsoluteDeviation",
            [x],
            np.asarray(st.median_abs_deviation(x, scale="normal")),
            scale="normal",
        ),
    ]
    return out
