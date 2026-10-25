"""Vendored regression datasets against their sources: Anscombe's statistics, mtcars and the CDC percentiles."""

from typing import cast

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.vendor import csv_rows, fetch_pinned
from scipy.stats import norm
from sklearn.datasets import load_linnerud
from sklearn.utils import Bunch

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}
ANSCOMBE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/anscombe.csv"
ANSCOMBE_SHA256 = "02a87e7348a05d2fc235e79e6bd89a9e279110d9398e98e32975d82323eb535c"
MTCARS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
MTCARS_SHA256 = "4d6dfa9b029dbcc15333a6e3598803fae8ae7c4d08605040b1706e6367695058"
CHARTS = {
    "weight": (
        "https://www.cdc.gov/growthcharts/data/zscore/wtage.csv",
        "3406c9d125bcb69c062a9e84eb8c0209bfe9346542bdc1d308643750dcc241b7",
    ),
    "stature": (
        "https://www.cdc.gov/growthcharts/data/zscore/statage.csv",
        "45130d2a9d7c50c54a47e7ba626b66c61d4554bc2d901198cedd9419a53f7251",
    ),
}


def cases() -> list[Case]:
    out: list[Case] = []
    rows = csv_rows(fetch_pinned(ANSCOMBE_URL, ANSCOMBE_SHA256))
    for s in range(1, 5):
        x = np.asarray([float(r[f"x{s}"]) for r in rows])
        y = np.asarray([float(r[f"y{s}"]) for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        stats = np.asarray(
            [x.mean(), y.mean(), x.var(ddof=1), y.var(ddof=1), np.corrcoef(x, y)[0, 1], slope, intercept]
        )
        out.append(case(f"anscombe/{s}", "anscombe", [], stats, rtol=1e-10, atol=1e-12, set=s))
    cars = csv_rows(fetch_pinned(MTCARS_URL, MTCARS_SHA256))
    mpg = np.asarray([float(r["mpg"]) for r in cars])
    out.append(case("mtcars/mpg", "mtcars", [], mpg, rtol=0, atol=0))
    # The CDC files tabulate percentiles beside the LMS parameters; the LMS formula must reproduce them.
    for measure, (url, sha) in CHARTS.items():
        chart = [r for r in csv_rows(fetch_pinned(url, sha)) if r["Sex"] == "2"]
        ages = np.asarray([float(r["Agemos"]) for r in chart[::20]])
        for p, column in [(0.05, "P5"), (0.5, "P50"), (0.95, "P95")]:
            expected = np.asarray([float(r[column]) for r in chart[::20]])
            out.append(
                case(
                    f"growth/{measure}/female/{column}",
                    "growthValue",
                    [ages],
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    measure=measure,
                    z=float(norm.ppf(p)),
                )
            )
    lin = cast(Bunch, load_linnerud())
    out.append(case("linnerud/x", "linnerud", [], np.asarray(lin.data), rtol=0, atol=0, part="x"))
    out.append(case("linnerud/y", "linnerud", [], np.asarray(lin.target), rtol=0, atol=0, part="y"))
    return out
