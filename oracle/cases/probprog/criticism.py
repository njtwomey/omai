"""Model criticism (probprog/criticism.ts): PSIS-LOO and model comparison with stacking weights against ArviZ
(arviz-stats `loo` and `compare`), and WAIC from its definition with numpy and scipy (arviz-stats 1.x has no WAIC)."""

import importlib.metadata
from typing import Any

import arviz_base as azb
import arviz_stats as azs
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.special import logsumexp

REFERENCE: dict[str, str] = {
    "arviz-stats": importlib.metadata.version("arviz-stats"),
    "arviz-base": importlib.metadata.version("arviz-base"),
    "numpy": np.__version__,
}

azs_any: Any = azs
azb_any: Any = azb


def tree(ll: FloatArray) -> object:
    """A DataTree with one chain of draws and the pointwise log-likelihoods `[draws, n]`."""
    s = ll.shape[0]
    return azb_any.from_dict({"posterior": {"mu": np.zeros((1, s))}, "log_likelihood": {"y": ll[None, :, :]}})


def waic(ll: FloatArray) -> tuple[float, float, float, FloatArray]:
    """elpd_waic, its standard error, p_waic and the pointwise elpd (Vehtari, Gelman and Gabry 2017)."""
    s = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(s)
    p_i = np.var(ll, axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    se = float(np.sqrt(len(elpd_i) * np.var(elpd_i, ddof=1)))
    return float(elpd_i.sum()), se, float(p_i.sum()), elpd_i


def cases() -> list[Case]:
    g = rng(31)
    out: list[Case] = []
    mats: dict[str, FloatArray] = {
        "tight": g.normal(-1.0, 0.2, size=(400, 6)),
        "wide": g.normal(-2.0, 0.8, size=(300, 5)),
        "outlier": np.concatenate([g.normal(-1.0, 0.3, size=(500, 4)), g.normal(-6.0, 2.0, size=(500, 1))], axis=1),
    }
    for name, ll in mats.items():
        elpd, se, p, pointwise = waic(ll)
        out.append(case(f"waic/{name}", "waic", [ll], np.array([elpd, se, p]), rtol=1e-12))
        out.append(case(f"waic/{name}/pointwise", "waicPointwise", [ll], pointwise, rtol=1e-12))
        r: Any = azs_any.loo(tree(ll), pointwise=True, reff=1.0)
        out.append(
            case(f"loo/{name}", "loo", [ll], np.array([float(r.elpd), float(r.se), float(r.p)]), rtol=1e-10, atol=1e-12)
        )
        out.append(case(f"loo/{name}/k", "looK", [ll], np.asarray(r.pareto_k, dtype=np.float64).ravel(), rtol=1e-10))
    a = mats["tight"]
    # Each model predicts some observations best, so the stacking weights are interior.
    b = a + g.normal(0.0, 0.1, size=a.shape) + np.array([-0.8, -0.8, -0.8, 0.6, 0.6, 0.6])
    c = a + g.normal(0.0, 0.1, size=a.shape) + np.array([0.5, -0.9, 0.4, -0.9, 0.3, -0.7])
    loos = {n: azs_any.loo(tree(m), pointwise=True, reff=1.0) for n, m in [("a", a), ("b", b), ("c", c)]}
    table: Any = azs_any.compare(loos, round_to="none")
    names = ["a", "b", "c"]
    out.append(
        case(
            "compare/stacking",
            "compare",
            [a, b, c],
            {
                "weight": np.array([float(table.loc[n, "weight"]) for n in names]),
                "elpdDiff": np.array([float(table.loc[n, "elpd_diff"]) for n in names]),
                "dse": np.array([float(table.loc[n, "dse"]) for n in names]),
                "rank": np.array([int(table.loc[n, "rank"]) for n in names], dtype=np.int32),
            },
            # ArviZ's stacking stops at scipy's SLSQP tolerance (ftol 1e-6 on the log score), so its weights are
            # accurate to about 1e-3; the differences and their standard errors are exact.
            rtol=1e-3,
            atol=1e-3,
        )
    )
    return out
