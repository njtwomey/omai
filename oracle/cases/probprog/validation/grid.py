"""The grid posterior (probprog/validation/grid.ts) against scipy's adaptive two-dimensional quadrature: the evidence
and the posterior means and standard deviations of a logistic regression with two coefficients, integrated independently
of the grid's own trapezoid rule."""

import importlib.metadata

import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": importlib.metadata.version("scipy")}

X = np.linspace(-1.5, 1.5, 16)
Y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=float)


def unnormalised(b1: float, b0: float) -> float:
    """exp of the log joint at (b0, b1): N(0, 2) priors and Bernoulli logits b0 + b1 x."""
    logits = b0 + b1 * X
    loglik = float(np.sum(Y * logits - np.logaddexp(0.0, logits)))
    return float(np.exp(loglik + st.norm(0, 2).logpdf(b0) + st.norm(0, 2).logpdf(b1)))


def cases() -> list[Case]:
    box = (-12.0, 12.0)
    z, _ = integrate.dblquad(unnormalised, *box, *box, epsabs=1e-13, epsrel=1e-12)
    moments: list[float] = [float(np.log(z))]
    for k in range(2):
        first, _ = integrate.dblquad(
            lambda b1, b0, k=k: (b0, b1)[k] * unnormalised(b1, b0), *box, *box, epsabs=1e-13, epsrel=1e-11
        )
        second, _ = integrate.dblquad(
            lambda b1, b0, k=k: (b0, b1)[k] ** 2 * unnormalised(b1, b0), *box, *box, epsabs=1e-13, epsrel=1e-11
        )
        mean = first / z
        moments.extend([mean, float(np.sqrt(second / z - mean**2))])
    return [case("logistic2", "logisticGrid", [X, Y], np.array(moments), rtol=1e-6, atol=1e-8)]
