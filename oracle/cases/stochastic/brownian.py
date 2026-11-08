"""Brownian motion's closed forms against scipy.stats: the running maximum by the reflection principle (halfnorm for no
drift, the inverse Gaussian survival for drift towards the level), first-passage densities and CDFs (invgauss and
levy) and the bridge's marginal moments (scipy.stats.norm of the conditional Gaussian)."""

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}


def invgauss(level: float, drift: float, sigma: float) -> tuple[float, float]:
    """scipy's (mu, scale) of the first-passage law to `level` > 0 with drift > 0: mean level / drift, shape
    level^2 / sigma^2."""
    shape = level**2 / sigma**2
    return (level / drift) / shape, shape


def cases() -> list[Case]:
    out: list[Case] = []
    t = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    for level, drift, sigma in [(1.0, 0.5, 1.0), (0.5, 2.0, 0.7), (2.0, 0.1, 1.5)]:
        mu, scale = invgauss(level, drift, sigma)
        law = st.invgauss(mu, scale=scale)
        params = {"level": level, "drift": drift, "sigma": sigma}
        out.append(case(f"fpt-pdf/{level}/{drift}", "fpt-pdf", [t], law.pdf(t), rtol=1e-10, atol=1e-14, **params))
        out.append(case(f"fpt-cdf/{level}/{drift}", "fpt-cdf", [t], law.cdf(t), rtol=1e-10, atol=1e-14, **params))
        out.append(case(f"max-cdf/{level}/{drift}", "max-cdf", [t], law.sf(t), rtol=1e-10, atol=1e-14, **params))
        out.append(
            case(f"fpt-dist-logpdf/{level}/{drift}", "fpt-dist-logpdf", [t], law.logpdf(t), rtol=1e-10, **params)
        )
    for level, sigma in [(1.0, 1.0), (-0.8, 0.6)]:
        law = st.levy(scale=level**2 / sigma**2)
        params = {"level": level, "drift": 0.0, "sigma": sigma}
        out.append(case(f"levy-pdf/{level}", "fpt-pdf", [t], law.pdf(t), rtol=1e-10, atol=1e-14, **params))
        out.append(case(f"levy-cdf/{level}", "fpt-cdf", [t], law.cdf(t), rtol=1e-10, atol=1e-14, **params))
        out.append(case(f"levy-logpdf/{level}", "fpt-dist-logpdf", [t], law.logpdf(t), rtol=1e-10, **params))
    m = np.array([0.0, 0.3, 1.0, 2.5])
    for horizon, sigma in [(1.0, 1.0), (3.0, 0.5)]:
        out.append(
            case(
                f"max-cdf-driftless/{horizon}",
                "max-cdf-m",
                [m],
                st.halfnorm(scale=sigma * np.sqrt(horizon)).cdf(m),
                rtol=1e-10,
                atol=1e-14,
                t=horizon,
                drift=0.0,
                sigma=sigma,
            )
        )
    s = np.array([0.1, 0.4, 0.7, 0.95])
    t0, t1, x0, x1, sigma = 0.0, 1.0, 0.5, -1.0, 1.3
    var = sigma**2 * (s - t0) * (t1 - s) / (t1 - t0)
    mean = x0 + (x1 - x0) * (s - t0) / (t1 - t0)
    params = {"t0": t0, "t1": t1, "x0": x0, "x1": x1, "sigma": sigma}
    out.append(case("bridge-mean", "bridge-mean", [s], st.norm(mean, np.sqrt(var)).mean(), **params))
    out.append(case("bridge-var", "bridge-var", [s], st.norm(mean, np.sqrt(var)).var(), **params))
    return out
