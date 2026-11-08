"""Panel designs against statsmodels and scipy: the two-by-two difference in differences and the two-way fixed-effects
regression as dummy-variable OLS with cluster-robust errors by unit, an event study as the same regression on
relative-time dummies, and synthetic control as the simplex-constrained least-squares problem solved by scipy's
SLSQP."""

import numpy as np
import numpy.typing as npt
import scipy
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.optimize import minimize

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__, "scipy": scipy.__version__}


def panel(
    seed: int, units: int, periods: int, staggered: bool
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """A long panel: unit, period, treated, outcome and the adoption period per unit (-1 for never)."""
    g = rng(seed)
    adoption = np.full(units, -1)
    treated_units = g.uniform(size=units) < 0.5
    for i in np.flatnonzero(treated_units):
        adoption[i] = int(g.integers(periods // 3, 2 * periods // 3)) if staggered else periods // 2
    unit_effect = g.normal(size=units)
    period_effect = g.normal(size=periods) * 0.5
    rows: list[tuple[float, float, float, float]] = []
    for i in range(units):
        for t in range(periods):
            d = 1.0 if adoption[i] >= 0 and t >= adoption[i] else 0.0
            tau = 0.0 if adoption[i] < 0 else 1.0 + 0.5 * max(0, t - adoption[i])
            rows.append((float(i), float(t), d, unit_effect[i] + period_effect[t] + d * tau + 0.4 * g.normal()))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], adoption.astype(np.float64)


def dummies(codes: FloatArray) -> FloatArray:
    levels = np.unique(codes)
    return np.asarray((codes[:, None] == levels[None, 1:]).astype(np.float64))


def did_case(name: str, seed: int, units: int, periods: int) -> Case:
    u, t, d, y, adoption = panel(seed, units, periods, False)
    cutoff = periods // 2
    ever = (adoption[u.astype(int)] >= 0).astype(np.float64)
    post = (t >= cutoff).astype(np.float64)
    design = sm.add_constant(np.column_stack([ever, post, ever * post]))  # pyright: ignore[reportUnknownMemberType]
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": u})  # pyright: ignore[reportUnknownMemberType]
    return case(
        name,
        "did",
        [u, t, d, y],
        {"value": np.array(float(fit.params[3])), "se": np.array(float(fit.bse[3]))},
        rtol=1e-7,
        atol=1e-9,
    )


def twfe_case(name: str, seed: int, units: int, periods: int, staggered: bool) -> Case:
    u, t, d, y, _ = panel(seed, units, periods, staggered)
    design = np.column_stack([d, dummies(u), dummies(t)])
    design = sm.add_constant(design)  # pyright: ignore[reportUnknownMemberType]
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": u})  # pyright: ignore[reportUnknownMemberType]
    return case(
        name,
        "twfe",
        [u, t, d, y],
        {"value": np.array(float(fit.params[1])), "se": np.array(float(fit.bse[1]))},
        rtol=1e-7,
        atol=1e-9,
    )


def event_case(name: str, seed: int, units: int, periods: int, leads: int, lags: int) -> Case:
    u, t, d, y, adoption = panel(seed, units, periods, True)
    adopt = adoption[u.astype(int)]
    rel = np.where(adopt >= 0, t - adopt, np.nan)
    times = sorted({int(v) for v in rel[~np.isnan(rel)]} - {-1})
    cols = np.column_stack([(rel == e).astype(np.float64) for e in times])
    design = sm.add_constant(np.column_stack([cols, dummies(u), dummies(t)]))  # pyright: ignore[reportUnknownMemberType]
    fit = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": u})  # pyright: ignore[reportUnknownMemberType]
    window = list(range(-leads, lags + 1))
    coef = [0.0 if e == -1 else float(fit.params[1 + times.index(e)]) for e in window]
    se = [0.0 if e == -1 else float(fit.bse[1 + times.index(e)]) for e in window]
    return case(
        name,
        "eventStudy",
        [u, t, d, y],
        {"coefficients": np.array(coef), "standardErrors": np.array(se)},
        rtol=1e-6,
        atol=1e-8,
        leads=leads,
        lags=lags,
    )


def synthetic_case(name: str, seed: int, donors: int, periods: int, start: int) -> Case:
    g = rng(seed)
    factors = np.cumsum(g.normal(size=(periods, 2)), axis=0) * 0.5
    loadings = g.normal(size=(donors + 1, 2))
    wide = factors @ loadings.T + 0.05 * g.normal(size=(periods, donors + 1))
    wide[start:, 0] += 2.0
    treated = wide[:, 0]
    pool = wide[:, 1:]
    pre = slice(0, start)

    def loss(w: npt.NDArray[np.float64]) -> float:
        r = treated[pre] - pool[pre] @ w
        return float(r @ r)

    res = minimize(  # pyright: ignore[reportUnknownVariableType]
        loss,
        np.full(donors, 1.0 / donors),
        jac=lambda w: -2 * pool[pre].T @ (treated[pre] - pool[pre] @ w),  # pyright: ignore[reportUnknownLambdaType]
        method="SLSQP",
        bounds=[(0.0, 1.0)] * donors,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],  # pyright: ignore[reportUnknownLambdaType]
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    w = np.asarray(res.x, dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
    gap = treated - pool @ w
    return case(
        name,
        "syntheticControl",
        [treated, pool],
        {"weights": w, "gap": np.asarray(gap, dtype=np.float64), "loss": np.array(loss(w))},
        rtol=1e-3,
        atol=1e-4,
        start=start,
    )


def cases() -> list[Case]:
    return [
        did_case("did/small", 201, 40, 6),
        did_case("did/large", 202, 120, 10),
        twfe_case("twfe/simultaneous", 211, 40, 6, False),
        twfe_case("twfe/staggered", 212, 60, 9, True),
        event_case("event/a", 221, 60, 10, 3, 3),
        event_case("event/b", 222, 80, 12, 2, 4),
        synthetic_case("synthetic/a", 231, 12, 24, 16),
        synthetic_case("synthetic/b", 232, 25, 30, 20),
    ]
