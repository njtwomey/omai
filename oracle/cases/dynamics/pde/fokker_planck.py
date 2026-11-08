"""Fokker-Planck densities against scipy.stats: the Ornstein-Uhlenbeck process dX = -theta X dt + sigma dW.

The stationary density is N(0, sigma^2 / (2 theta)); from a Gaussian start N(m0, v0) the density stays Gaussian with
mean m0 e^{-theta t} and variance sigma^2 / (2 theta) (1 - e^{-2 theta t}) + v0 e^{-2 theta t}. Compared on the
cells at the discretisation's accuracy.
"""

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for theta, sigma in [(1.0, 1.0), (2.0, 0.5)]:
        var = sigma**2 / (2 * theta)
        x = np.linspace(-6 * np.sqrt(var), 6 * np.sqrt(var), 201)
        out.append(
            case(
                f"stationary/{theta}/{sigma}",
                "stationary",
                [x],
                st.norm.pdf(x, scale=np.sqrt(var)),
                rtol=1e-4,
                atol=1e-6,
                theta=theta,
                sigma=sigma,
            )
        )
        m0, v0, t = 1.0, 0.05, 0.5
        mean = m0 * np.exp(-theta * t)
        variance = var * (1 - np.exp(-2 * theta * t)) + v0 * np.exp(-2 * theta * t)
        cells = 400
        lo, hi = float(-6 * np.sqrt(var) - 1), float(6 * np.sqrt(var) + 1)
        dx = (hi - lo) / cells
        centres = lo + (np.arange(cells) + 0.5) * dx
        out.append(
            case(
                f"transient/{theta}/{sigma}",
                "transient",
                [],
                st.norm.pdf(centres, loc=mean, scale=np.sqrt(variance)),
                rtol=0,
                atol=2e-2,
                theta=theta,
                sigma=sigma,
                lo=lo,
                hi=hi,
                cells=cells,
                m0=m0,
                v0=v0,
                t=t,
            )
        )
    return out
