"""Point processes: Ripley's K with every edge correction of astropy.stats.RipleysKEstimator, the exponential Hawkes
log-likelihood (closed form in numpy) and its maximum-likelihood estimate (scipy.optimize.minimize, L-BFGS-B, on the
log-parameters), and the renewal function of gamma(2) gaps (closed form)."""

from typing import Literal

import astropy
import numpy as np
import scipy
import scipy.optimize as so
from astropy.stats import RipleysKEstimator
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"astropy": astropy.__version__, "scipy": scipy.__version__, "numpy": np.__version__}


def hawkes_ll(params: FloatArray, events: FloatArray, horizon: float) -> float:
    mu, alpha, beta = params
    lags = events[:, None] - events[None, :]
    kernel = np.where(lags > 0, np.exp(-beta * np.where(lags > 0, lags, 0.0)), 0.0)
    lam = mu + alpha * kernel.sum(axis=1)
    tail = np.sum(1 - np.exp(-beta * (horizon - events)))
    return float(np.sum(np.log(lam)) - mu * horizon - alpha / beta * tail)


def hawkes_events(g: np.random.Generator, mu: float, alpha: float, beta: float, horizon: float) -> FloatArray:
    """Ogata thinning, to produce a data set (the fixture checks the likelihood and its maximiser, not the sampler)."""
    t, excitation, out = 0.0, 0.0, []
    while True:
        bound = mu + excitation
        nxt = t + g.exponential(1 / bound)
        if nxt > horizon:
            return np.array(out)
        excitation *= np.exp(-beta * (nxt - t))
        t = nxt
        if g.uniform() * bound <= mu + excitation:
            out.append(t)
            excitation += alpha


def cases() -> list[Case]:
    g = rng(31)
    out: list[Case] = []
    pts = np.column_stack([g.uniform(0, 2, size=30), g.uniform(0, 1, size=30)])
    radii = np.array([0.05, 0.1, 0.2, 0.3, 0.45])
    est = RipleysKEstimator(area=2.0, x_min=0, x_max=2, y_min=0, y_max=1)
    window = {"xmin": 0.0, "xmax": 2.0, "ymin": 0.0, "ymax": 1.0}
    modes: list[Literal["none", "translation", "ohser", "var-width", "ripley"]] = [
        "none",
        "translation",
        "ohser",
        "var-width",
        "ripley",
    ]
    for mode in modes:
        k = est(data=pts, radii=radii, mode=mode)
        out.append(case(f"ripley/{mode}", "ripley", [pts, radii], k, rtol=1e-10, atol=1e-12, mode=mode, **window))
    events = hawkes_events(g, 0.8, 0.5, 1.2, 150.0)
    params = np.array([0.8, 0.5, 1.2])
    out.append(
        case(
            "hawkes/ll",
            "hawkes-ll",
            [events, params],
            np.array(hawkes_ll(params, events, 150.0)),
            rtol=1e-11,
            horizon=150.0,
        )
    )

    def nll(logp: FloatArray) -> float:
        return -hawkes_ll(np.exp(logp), events, 150.0)

    res = so.minimize(nll, np.log([0.5, 0.3, 1.0]), method="L-BFGS-B", options={"gtol": 1e-12, "ftol": 1e-15})
    out.append(case("hawkes/mle", "hawkes-mle", [events], np.exp(res.x), rtol=1e-5, horizon=150.0))
    lam = 1.5
    t = np.array([0.5, 1.0, 2.0, 5.0])
    renewal = lam * t / 2 - 0.25 * (1 - np.exp(-2 * lam * t))
    # The renewal equation is solved on a grid of 2000 steps: second-order accurate, error about 1e-6.
    out.append(case("renewal/gamma2", "renewal-gamma", [t], renewal, rtol=1e-5, atol=1e-6, rate=lam))
    return out
