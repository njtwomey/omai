"""Correlation and agreement measures against scipy.stats, torchmetrics (concordance), pingouin (ICC), statsmodels
(Fleiss' kappa), sklearn (weighted kappa) and the krippendorff package; Bland-Altman and Theil's U from their
definitions.
"""

from importlib.metadata import version
from typing import Literal

import krippendorff
import numpy as np
import pandas as pd
import pingouin as pg
import scipy
import scipy.stats as st
import sklearn.metrics as M
import torchmetrics.functional.regression as TR
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from scipy.stats.contingency import association, crosstab
from statsmodels.stats.inter_rater import fleiss_kappa

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "pingouin": pg.__version__,
    "krippendorff": version("krippendorff"),
    "torchmetrics": version("torchmetrics"),
}


def theils_u(x: FloatArray, y: FloatArray) -> float:
    table = np.asarray(crosstab(x, y).count, dtype=np.float64)
    hx = float(st.entropy(table.sum(1)))
    h_given = sum(table[:, j].sum() / table.sum() * float(st.entropy(table[:, j])) for j in range(table.shape[1]))
    return (hx - h_given) / hx


def icc(ratings: FloatArray, kind: str) -> float:
    n, k = ratings.shape
    frame = pd.DataFrame(
        {"target": np.repeat(np.arange(n), k), "rater": np.tile(np.arange(k), n), "rating": ratings.ravel()}
    )
    table = pg.intraclass_corr(data=frame, targets="target", raters="rater", ratings="rating")
    names = {
        "ICC1": "ICC(1,1)",
        "ICC2": "ICC(A,1)",
        "ICC3": "ICC(C,1)",
        "ICC1k": "ICC(1,k)",
        "ICC2k": "ICC(A,k)",
        "ICC3k": "ICC(C,k)",
    }
    return float(table.set_index("Type").loc[names[kind], "ICC"])


def cases() -> list[Case]:
    g = rng(65)
    n = 30
    x = g.normal(size=n)
    y = 0.6 * x + 0.8 * g.normal(size=n)
    yt = np.round(y, 0)
    w = g.uniform(0.3, 2.0, size=n)
    xc = g.integers(0, 3, size=n).astype(np.float64)
    yc = np.where(g.uniform(size=n) < 0.6, xc, g.integers(0, 4, size=n)).astype(np.float64)
    ratings = g.normal(size=(12, 1)) + 0.5 * g.normal(size=(12, 4)) + np.array([0.0, 0.3, -0.2, 0.5])
    counts = g.multinomial(6, [0.5, 0.3, 0.2], size=10).astype(np.float64)
    rel = g.integers(1, 5, size=(4, 15)).astype(np.float64)
    rel[g.uniform(size=(4, 15)) < 0.2] = np.nan
    weighted = np.cov(np.stack([x, y]), aweights=w)
    levels: list[Literal["nominal", "ordinal", "interval", "ratio"]] = ["nominal", "ordinal", "interval", "ratio"]
    out: list[Case] = [
        case("pearsonCorrelation", "pearsonCorrelation", [x, y], np.asarray(st.pearsonr(x, y).statistic)),
        case(
            "pearsonCorrelation/weighted",
            "pearsonCorrelation",
            [x, y, w],
            np.asarray(weighted[0, 1] / np.sqrt(weighted[0, 0] * weighted[1, 1])),
        ),
        case("spearmanCorrelation", "spearmanCorrelation", [x, yt], np.asarray(st.spearmanr(x, yt).statistic)),
        case("kendallTau/b", "kendallTau", [x, yt], np.asarray(st.kendalltau(x, yt).statistic)),
        case(
            "kendallTau/c",
            "kendallTau",
            [np.round(x), yt],
            np.asarray(st.kendalltau(np.round(x), yt, variant="c").statistic),
            variant="c",
        ),
        case(
            "concordanceCorrelation",
            "concordanceCorrelation",
            [x, y],
            to_numpy(TR.concordance_corrcoef(to_torch(y), to_torch(x))).reshape(()),
            rtol=1e-10,
        ),
        case(
            "concordanceCorrelation/lin",
            "concordanceCorrelation",
            [x, y],
            np.asarray(2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)),
            ddof=0,
        ),
        case(
            "blandAltman/lower",
            "blandAltmanLower",
            [x, y],
            np.asarray((x - y).mean() - st.norm.ppf(0.975) * (x - y).std(ddof=1)),
        ),
        case(
            "blandAltman/upper",
            "blandAltmanUpper",
            [x, y],
            np.asarray((x - y).mean() + st.norm.ppf(0.975) * (x - y).std(ddof=1)),
        ),
        case(
            "cramersV",
            "cramersV",
            [xc, yc],
            np.asarray(association(np.asarray(crosstab(xc, yc).count), method="cramer", correction=False)),
        ),
        case("theilsU", "theilsU", [xc, yc], np.asarray(theils_u(xc, yc))),
        case(
            "weightedCohenKappa",
            "weightedCohenKappa",
            [xc, yc],
            np.asarray(M.cohen_kappa_score(xc, yc, weights="linear")),
        ),
        case("fleissKappa", "fleissKappa", [counts, counts], np.asarray(fleiss_kappa(counts))),
        case(
            "fleissKappa/randolph",
            "fleissKappa",
            [counts, counts],
            np.asarray(fleiss_kappa(counts, method="randolph")),
            method="randolph",
        ),
    ]
    for kind in ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]:
        out.append(
            case(
                f"intraclassCorrelation/{kind}",
                "intraclassCorrelation",
                [ratings, ratings],
                np.asarray(icc(ratings, kind)),
                type=kind,
                rtol=1e-10,
            )
        )
    for level in levels:
        out.append(
            case(
                f"krippendorffAlpha/{level}",
                "krippendorffAlpha",
                [rel, rel],
                np.asarray(krippendorff.alpha(reliability_data=rel, level_of_measurement=level)),
                level=level,
                rtol=1e-10,
            )
        )
    return out
