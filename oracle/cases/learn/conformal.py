"""Conformal prediction against MAPIE: split conformal regression (absolute and gamma scores), conformalized quantile
regression (symmetric and not), CV+ and the jackknife+ (and jackknife-minmax), Mondrian (per-group) intervals, and
split conformal classification with LAC and APS scores (APS calibration randomised with the same uniforms)."""

from typing import Any

import mapie
import numpy as np
from mapie.classification import SplitConformalClassifier
from mapie.regression import ConformalizedQuantileRegressor, CrossConformalRegressor, SplitConformalRegressor
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"mapie": mapie.__version__}


def bounds(intervals: Any) -> tuple[FloatArray, FloatArray]:  # noqa: ANN401  (MAPIE returns untyped tuples)
    """The lower and upper bounds of MAPIE's (n, 2, 1) interval array."""
    arr = np.asarray(intervals)
    return arr[:, 0, 0], arr[:, 1, 0]


def cases() -> list[Case]:
    g = rng(4342)
    n = 80
    x = g.normal(size=(n, 2))
    y = x @ np.array([1.5, -1.0]) + (0.5 + 0.5 * np.abs(x[:, 0])) * g.normal(size=n)
    xt = g.normal(size=(6, 2))
    tr, cal = np.arange(40), np.arange(40, n)
    out: list[Case] = []
    lr = LinearRegression().fit(x[tr], y[tr])
    for level in [0.8, 0.9]:
        scr = SplitConformalRegressor(lr, confidence_level=level, prefit=True).conformalize(x[cal], y[cal])
        lo, hi = bounds(scr.predict_interval(xt)[1])
        out.append(
            case(
                f"split/absolute/{level}",
                "splitAbsolute",
                [x[tr], y[tr], x[cal], y[cal], xt],
                {"lower": lo, "upper": hi},
                alpha=1 - level,
                rtol=1e-10,
                atol=1e-10,
            )
        )
    ypos = np.exp(0.3 * y)
    lrp = LinearRegression().fit(x[tr], ypos[tr])
    scr = SplitConformalRegressor(lrp, confidence_level=0.9, conformity_score="gamma", prefit=True)
    scr.conformalize(x[cal], ypos[cal])
    xt_pos = xt * 0.2
    lo, hi = bounds(scr.predict_interval(xt_pos)[1])
    out.append(
        case(
            "split/gamma",
            "splitGamma",
            [x[tr], ypos[tr], x[cal], ypos[cal], xt_pos],
            {"lower": lo, "upper": hi},
            alpha=0.1,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    # MAPIE types estimator as one regressor; a prefit [lower, upper, median] list is what it expects here.
    models: Any = [LinearRegression().fit(x[tr], y[tr] - 1.0), LinearRegression().fit(x[tr], y[tr] + 1.0), lr]
    for symmetric in [True, False]:
        cqr = ConformalizedQuantileRegressor(models, confidence_level=0.9, prefit=True).conformalize(x[cal], y[cal])
        lo, hi = bounds(cqr.predict_interval(xt, symmetric_correction=symmetric)[1])
        out.append(
            case(
                f"cqr/{symmetric}",
                "cqr",
                [x[tr], y[tr], x[cal], y[cal], xt],
                {"lower": lo, "upper": hi},
                alpha=0.1,
                symmetric=symmetric,
                rtol=1e-10,
                atol=1e-10,
            )
        )
    cv_settings: list[tuple[str, Any, str]] = [
        ("cv-plus", KFold(5), "plus"),
        ("jackknife-plus", LeaveOneOut(), "plus"),
        ("jackknife-minmax", LeaveOneOut(), "minmax"),
    ]
    for label, cv, method in cv_settings:
        ccr = CrossConformalRegressor(LinearRegression(), confidence_level=0.9, method=method, cv=cv)
        ccr.fit_conformalize(x[:30], y[:30])
        lo, hi = bounds(ccr.predict_interval(xt)[1])
        out.append(
            case(
                f"cross/{label}",
                "cross",
                [x[:30], y[:30], xt],
                {"lower": lo, "upper": hi},
                alpha=0.1,
                method=method,
                loo=label != "cv-plus",
                rtol=1e-10,
                atol=1e-10,
            )
        )
    groups = (x[:, 0] > 0).astype(np.float64)
    lo_all = np.zeros(len(xt))
    hi_all = np.zeros(len(xt))
    gt = (xt[:, 0] > 0).astype(np.float64)
    for grp in [0.0, 1.0]:
        sel = cal[groups[cal] == grp]
        scr = SplitConformalRegressor(lr, confidence_level=0.8, prefit=True).conformalize(x[sel], y[sel])
        lo, hi = bounds(scr.predict_interval(xt)[1])
        lo_all[gt == grp] = lo[gt == grp]
        hi_all[gt == grp] = hi[gt == grp]
    out.append(
        case(
            "mondrian",
            "mondrian",
            [x[tr], y[tr], x[cal], y[cal], xt, groups[cal], gt],
            {"lower": lo_all, "upper": hi_all},
            alpha=0.2,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    yc = np.digitize(x[:, 0] - 0.5 * x[:, 1] + 0.8 * g.normal(size=n), [-0.6, 0.6]).astype(np.float64)
    nb = GaussianNB().fit(x[tr], yc[tr])
    scc = SplitConformalClassifier(nb, confidence_level=0.8, conformity_score="lac", prefit=True)
    scc.conformalize(x[cal], yc[cal])
    sets = np.asarray(scc.predict_set(xt)[1])[:, :, 0].astype(np.float64)
    out.append(
        case(
            "classifier/lac", "classifier", [x[tr], yc[tr], x[cal], yc[cal], xt], {"sets": sets}, alpha=0.2, score="lac"
        )
    )
    seed = 7
    scc = SplitConformalClassifier(nb, confidence_level=0.8, conformity_score="aps", prefit=True, random_state=seed)
    scc.conformalize(x[cal], yc[cal])
    sets = np.asarray(scc.predict_set(xt, conformity_score_params={"include_last_label": True})[1])[:, :, 0]
    u = np.random.RandomState(seed).uniform(size=len(cal))
    out.append(
        case(
            "classifier/aps",
            "classifier",
            [x[tr], yc[tr], x[cal], yc[cal], xt, u],
            {"sets": sets.astype(np.float64)},
            alpha=0.2,
            score="aps",
        )
    )
    return out
