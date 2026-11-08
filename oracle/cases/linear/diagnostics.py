"""Regression diagnostics against statsmodels: OLSInfluence (leverage, studentised residuals, Cook's distance, DFFITS,
DFBETAS, COVRATIO), variance_inflation_factor, het_breuschpagan (Koenker and classical) and het_white; weighted fits
are the ordinary fit of the whitened data."""

import numpy as np
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(1980)
    out: list[Case] = []
    for name, n, d, weighted in [("small", 12, 2, False), ("wide", 30, 4, False), ("weighted", 20, 3, True)]:
        x = g.normal(size=(n, d))
        x[:, 1] += 0.7 * x[:, 0]  # correlated columns give non-trivial variance inflation
        y = x @ np.arange(1, d + 1) + 2.0 + (1 + 0.5 * np.abs(x[:, 0])) * g.normal(size=n)
        y[3] += 6.0  # one outlier
        w = g.uniform(0.5, 2.0, size=n) if weighted else np.ones(n)
        exog = sm.add_constant(x, prepend=False)
        # statsmodels' OLSInfluence on a WLS result mixes the raw and whitened design in the hat matrix; the
        # well-defined weighted diagnostics are those of the whitened ordinary fit, which is what is compared.
        sw = np.sqrt(w)
        res = sm.OLS(sw * y, sw[:, None] * exog).fit()
        infl = OLSInfluence(res)
        out.append(
            case(
                f"influence/{name}",
                "influence",
                [x, y, w],
                {
                    "leverage": infl.hat_matrix_diag,
                    "internal": infl.resid_studentized_internal,
                    "external": infl.resid_studentized_external,
                    "cooks": infl.cooks_distance[0],  # pyright: ignore[reportIndexIssue]
                    "dffits": infl.dffits[0],  # pyright: ignore[reportIndexIssue]
                    "dffitsInternal": infl.dffits_internal[0],  # pyright: ignore[reportIndexIssue]
                    "dfbeta": infl.dfbeta,
                    "dfbetas": infl.dfbetas,
                    "covRatio": infl.cov_ratio,
                },
                rtol=1e-9,
                atol=1e-10,
            )
        )
        if not weighted:
            vif = np.array([variance_inflation_factor(exog, j) for j in range(d)])
            out.append(case(f"vif/{name}", "vif", [x], {"vif": vif}, rtol=1e-9, atol=1e-10))
            resid = res.resid
            lm, lm_p, _, _ = het_breuschpagan(resid, exog)
            lm0, lm0_p, _, _ = het_breuschpagan(resid, exog, robust=False)
            wlm, wp, _, _ = het_white(resid, exog)
            out.append(
                case(
                    f"het/{name}",
                    "het",
                    [x, y],
                    {
                        "bp": np.array([lm, lm_p]),
                        "bpClassic": np.array([lm0, lm0_p]),
                        "white": np.array([wlm, wp]),
                    },
                    rtol=1e-8,
                    atol=1e-10,
                )
            )
    return out
