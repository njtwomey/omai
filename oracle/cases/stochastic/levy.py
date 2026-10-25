"""Levy processes: alpha-stable densities, CDFs and quantiles against scipy.stats.levy_stable (S1 parameterisation),
the variance gamma density against the normal variance-mean mixture integrated by scipy.integrate.quad, and the
variance gamma characteristic function (closed form, numpy complex powers)."""

import numpy as np
import scipy
import scipy.stats as st
from mdml_oracle.fixture import Case, case
from scipy.integrate import quad

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}

STABLE = [
    (1.5, 0.0, 1.0, 0.0),
    (1.5, 0.5, 2.0, 1.0),
    (0.8, 0.3, 1.0, 0.0),
    (1.9, -0.7, 0.5, -1.0),
    (1.2, 1.0, 1.0, 0.0),
]


def vg_density(x: float, t: float, theta: float, sigma: float, nu: float) -> float:
    def integrand(g: float) -> float:
        return float(st.norm.pdf(x, theta * g, sigma * np.sqrt(g)) * st.gamma.pdf(g, t / nu, scale=nu))

    value, _ = quad(integrand, 0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=200)
    return float(value)


def cases() -> list[Case]:
    out: list[Case] = []
    xs = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for alpha, beta, scale, loc in STABLE:
        law = st.levy_stable(alpha, beta, loc=loc, scale=scale)
        params = {"alpha": alpha, "beta": beta, "scale": scale, "loc": loc}
        tag = f"{alpha}/{beta}"
        out.append(case(f"stable-pdf/{tag}", "stable-pdf", [xs], law.pdf(xs), rtol=1e-6, atol=1e-9, **params))
        out.append(case(f"stable-cdf/{tag}", "stable-cdf", [xs], law.cdf(xs), rtol=1e-6, atol=1e-9, **params))
        out.append(case(f"stable-ppf/{tag}", "stable-quantiles", [probs], law.ppf(probs), rtol=1e-6, **params))
    for t, theta, sigma, nu in [(1.0, -0.2, 0.3, 0.4), (0.5, 0.1, 0.5, 0.2)]:
        x = np.array([-0.8, -0.3, 0.05, 0.2, 0.7])
        dens = np.array([vg_density(float(v), t, theta, sigma, nu) for v in x])
        p = {"t": t, "theta": theta, "sigma": sigma, "nu": nu}
        out.append(case(f"vg-density/{t}", "vg-density", [x], dens, rtol=1e-8, atol=1e-12, **p))
        u = np.array([0.0, 0.5, 1.0, 3.0])
        cf = (1 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2) ** (-t / nu)
        out.append(case(f"vg-cf-re/{t}", "vg-cf-re", [u], np.real(cf), rtol=1e-12, atol=1e-15, **p))
        out.append(case(f"vg-cf-im/{t}", "vg-cf-im", [u], np.imag(cf), rtol=1e-12, atol=1e-15, **p))
    return out
