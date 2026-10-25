"""Feature selection against sklearn.feature_selection: univariate scores (F tests, correlation, chi-squared, mutual
information), the variance threshold, k-best, percentile and p-value selectors, selection from model importances,
RFE, RFECV and sequential selection."""

from typing import Any

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def support(selector: Any) -> np.ndarray[Any, np.dtype[np.float64]]:  # noqa: ANN401  (sklearn selectors are untyped)
    """The support mask as floats."""
    return np.asarray(selector.get_support(), dtype=np.float64)


def cases() -> list[Case]:
    g = rng(4340)
    n, d = 50, 6
    x = g.normal(size=(n, d))
    x[:, 4] = 0.5 * x[:, 0] + 0.1 * g.normal(size=n)
    y = 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] + 0.5 * g.normal(size=n)
    yc = (x[:, 0] - 0.5 * x[:, 3] + 0.5 * g.normal(size=n) > 0).astype(np.float64)
    counts = g.poisson(lam=np.where(yc[:, None] > 0, [3, 1, 2, 2], [1, 3, 2, 2]), size=(n, 4)).astype(np.float64)
    out: list[Case] = []
    f, p = f_classif(x, yc)
    out.append(case("fClassif", "fClassif", [x, yc], {"scores": f, "pValues": p}, rtol=1e-10, atol=1e-14))
    f, p = f_regression(x, y)
    out.append(case("fRegression", "fRegression", [x, y], {"scores": f, "pValues": p}, rtol=1e-10, atol=1e-14))
    out.append(case("rRegression", "rRegression", [x, y], {"scores": r_regression(x, y)}, rtol=1e-12))
    c, p = chi2(counts, yc)
    out.append(case("chi2", "chi2", [counts, yc], {"scores": c, "pValues": p}, rtol=1e-10, atol=1e-14))
    # sklearn jitters continuous features by 1e-10 of their scale; the estimates agree to that order.
    out.append(
        case(
            "mutualInfoClassif",
            "mutualInfoClassif",
            [x, yc],
            {"scores": mutual_info_classif(x, yc, random_state=0)},
            rtol=1e-6,
            atol=1e-8,
        )
    )
    out.append(
        case(
            "mutualInfoRegression",
            "mutualInfoRegression",
            [x, y],
            {"scores": mutual_info_regression(x, y, random_state=0)},
            rtol=1e-6,
            atol=1e-8,
        )
    )
    xv = np.column_stack([x[:, :3], np.full(n, 2.0), 0.3 * x[:, 3]])
    for threshold in [0.0, 0.5]:
        out.append(
            case(
                f"variance/{threshold}",
                "variance",
                [xv, y],
                {"support": support(VarianceThreshold(threshold).fit(xv))},
                threshold=threshold,
            )
        )
    out.append(case("kBest", "kBest", [x, y], {"support": support(SelectKBest(f_regression, k=2).fit(x, y))}, k=2))
    out.append(
        case(
            "percentile",
            "percentile",
            [x, y],
            {"support": support(SelectPercentile(f_regression, percentile=40).fit(x, y))},
            percentile=40,
        )
    )
    for name, sel in [
        ("fpr", SelectFpr(f_regression, alpha=0.05)),
        ("fdr", SelectFdr(f_regression, alpha=0.05)),
        ("fwe", SelectFwe(f_regression, alpha=0.05)),
    ]:
        out.append(case(f"pValue/{name}", "pValue", [x, y], {"support": support(sel.fit(x, y))}, control=name))
    for threshold in ["mean", "median", "1.5*mean"]:
        sel = SelectFromModel(Ridge(alpha=1.0), threshold=threshold).fit(x, y)
        out.append(case(f"fromModel/{threshold}", "fromModel", [x, y], {"support": support(sel)}, threshold=threshold))
    sel = SelectFromModel(Ridge(alpha=1.0), threshold=-np.inf, max_features=3).fit(x, y)
    out.append(case("fromModel/max", "fromModel", [x, y], {"support": support(sel)}, maxFeatures=3))
    rfe_settings: list[tuple[int, Any]] = [(2, 1), (3, 2), (1, 0.4)]
    for target, step in rfe_settings:
        rfe = RFE(LinearRegression(), n_features_to_select=target, step=step).fit(x, y)
        out.append(
            case(
                f"rfe/{target}-{step}",
                "rfe",
                [x, y],
                {"support": support(rfe), "ranking": rfe.ranking_.astype(np.float64)},
                nFeatures=target,
                step=step,
            )
        )
    rfecv = RFECV(LinearRegression(), cv=KFold(3), scoring="r2").fit(x, y)
    res: dict[str, Any] = rfecv.cv_results_
    out.append(
        case(
            "rfecv",
            "rfecv",
            [x, y],
            {
                "support": support(rfecv),
                "mean": res["mean_test_score"],
                "std": res["std_test_score"],
                "nFeatures": np.asarray(res["n_features"], dtype=np.float64),
            },
            rtol=1e-10,
            atol=1e-12,
        )
    )
    for direction in ["forward", "backward"]:
        # sklearn's signature types n_features_to_select as str and cv as int; both accept more.
        sfs_options: dict[str, Any] = {"n_features_to_select": 2, "direction": direction, "cv": KFold(3)}
        sfs = SequentialFeatureSelector(LinearRegression(), **sfs_options)
        out.append(
            case(
                f"sequential/{direction}",
                "sequential",
                [x, y],
                {"support": support(sfs.fit(x, y))},
                direction=direction,
            )
        )
    return out
