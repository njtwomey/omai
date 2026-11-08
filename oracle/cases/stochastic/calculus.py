"""The Black-Scholes price (the Feynman-Kac solution of the Black-Scholes PDE) against its closed form with
scipy.stats.norm, for calls and puts with and without a dividend yield."""

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}


def black_scholes(s: float, k: float, t: float, r: float, sigma: float, q: float, call: bool) -> float:
    """The closed-form European option price."""
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if call:
        return float(s * np.exp(-q * t) * st.norm.cdf(d1) - k * np.exp(-r * t) * st.norm.cdf(d2))
    return float(k * np.exp(-r * t) * st.norm.cdf(-d2) - s * np.exp(-q * t) * st.norm.cdf(-d1))


def cases() -> list[Case]:
    out: list[Case] = []
    grid = [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (90.0, 110.0, 0.5, 0.01, 0.35, 0.0),
        (120.0, 100.0, 2.0, 0.03, 0.15, 0.02),
        (50.0, 55.0, 0.1, 0.0, 0.5, 0.01),
    ]
    for s, k, t, r, sigma, q in grid:
        for kind in ["call", "put"]:
            price = black_scholes(s, k, t, r, sigma, q, kind == "call")
            out.append(
                case(
                    f"black-scholes/{kind}/{s}/{k}/{t}",
                    "black-scholes",
                    [],
                    np.array(price),
                    rtol=1e-10,
                    atol=1e-12,
                    spot=s,
                    strike=k,
                    maturity=t,
                    rate=r,
                    sigma=sigma,
                    dividend=q,
                    kind=kind,
                )
            )
    return out
