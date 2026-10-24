"""Linear regression methods against scikit-learn and statsmodels: the lasso path, ridge, the elastic net, LassoCV's
cross-validated errors, RidgeCV's leave-one-out choice, orthogonal matching pursuit and the OLS summary."""

import numpy as np
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, LassoCV, OrthogonalMatchingPursuit, Ridge, RidgeCV, lasso_path
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__}


def problem(seed: int, n: int = 60, d: int = 5) -> tuple[FloatArray, FloatArray]:
    """Correlated features and a sparse linear signal with noise."""
    g = rng(seed)
    z = g.normal(size=(n, d))
    x = z + 0.5 * z[:, [0]]
    beta = np.array([1.5, 0.0, -2.0, 0.0, 0.7])[:d]
    y = x @ beta + 0.3 + 0.5 * g.normal(size=n)
    return x, y


def cases() -> list[Case]:
    out: list[Case] = []
    x, y = problem(1)
    xc = x - x.mean(0)
    yc = y - y.mean()
    amax = float(np.max(np.abs(xc.T @ yc)) / len(y))
    alphas = amax * np.geomspace(1, 1e-3, 12)
    _, coefs, _ = lasso_path(xc, yc, alphas=alphas, tol=1e-12, max_iter=100000)  # pyright: ignore[reportArgumentType]
    out.append(case("lassoPath", "lassoPath", [x, y, alphas], {"coefs": coefs.T}, rtol=1e-7, atol=1e-8))
    r = Ridge(alpha=1.5).fit(x, y)
    out.append(
        case(
            "ridge",
            "ridge",
            [x, y],
            {"coef": r.coef_, "intercept": np.array(r.intercept_)},
            rtol=1e-10,
            atol=1e-10,
            alpha=1.5,
        )
    )
    w = rng(2).uniform(0.5, 2, size=len(y))
    r = Ridge(alpha=0.7).fit(x, y, sample_weight=w)
    out.append(
        case(
            "ridge/weighted",
            "ridge",
            [x, y, w],
            {"coef": r.coef_, "intercept": np.array(r.intercept_)},
            rtol=1e-10,
            atol=1e-10,
            alpha=0.7,
        )
    )
    e = ElasticNet(alpha=0.1, l1_ratio=0.7, tol=1e-12, max_iter=100000).fit(x, y)
    out.append(
        case(
            "elasticNet",
            "elasticNet",
            [x, y],
            {"coef": e.coef_, "intercept": np.array(e.intercept_)},
            rtol=1e-7,
            atol=1e-8,
            alpha=0.1,
            l1Ratio=0.7,
        )
    )
    cv = LassoCV(alphas=alphas, cv=KFold(5), tol=1e-12, max_iter=100000).fit(x, y)  # pyright: ignore[reportArgumentType]
    out.append(
        case(
            "lassoCV",
            "lassoCV",
            [x, y, alphas],
            {"mse": cv.mse_path_, "alpha": np.array(cv.alpha_), "coef": cv.coef_},
            rtol=1e-6,
            atol=1e-7,
        )
    )
    rcv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(x, y)
    out.append(
        case("ridgeCV", "ridgeCV", [x, y], {"alpha": np.array(rcv.alpha_), "coef": rcv.coef_}, rtol=1e-9, atol=1e-9)
    )
    o = OrthogonalMatchingPursuit(n_nonzero_coefs=2).fit(x, y)
    out.append(
        case("omp", "omp", [x, y], {"coef": o.coef_, "intercept": np.array(o.intercept_)}, rtol=1e-10, atol=1e-10, k=2)
    )
    exog = np.column_stack([x, np.ones(len(y))])
    ols = sm.OLS(y, exog).fit()
    hc3 = sm.OLS(y, exog).fit(cov_type="HC3")
    infl = ols.get_influence()
    vif = np.array([variance_inflation_factor(exog, j) for j in range(x.shape[1])])
    out.append(
        case(
            "olsSummary",
            "olsSummary",
            [x, y],
            {
                "params": ols.params,
                "bse": ols.bse,
                "pvalues": ols.pvalues,
                "hc3": hc3.bse,
                "summary": np.array(
                    [ols.rsquared, ols.rsquared_adj, ols.fvalue, ols.f_pvalue, ols.llf, ols.aic, ols.bic]
                ),
                "cooks": infl.cooks_distance[0],
                "leverage": infl.hat_matrix_diag,
                "vif": vif,
            },
            rtol=1e-8,
            atol=1e-10,
        )
    )
    return out
