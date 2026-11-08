"""Queues: Erlang B and C (the Poisson pmf over CDF ratio of scipy.stats.poisson), and M/M/1 and M/M/c measures from
the stationary law of the truncated birth-death generator (scipy.linalg.null_space), with the M/G/1
Pollaczek-Khinchine mean queue length for exponential, deterministic and uniform service."""

import numpy as np
import scipy
import scipy.linalg as sla
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def erlang_b(c: int, a: float) -> float:
    return float(st.poisson.pmf(c, a) / st.poisson.cdf(c, a))


def mmc_measures(lam: float, mu: float, c: int, size: int = 600) -> FloatArray:
    """[L, Lq, p0] from the stationary law of the M/M/c birth-death chain truncated at `size`."""
    q = np.zeros((size + 1, size + 1))
    for n in range(size + 1):
        if n < size:
            q[n, n + 1] = lam
        if n > 0:
            q[n, n - 1] = mu * min(n, c)
        q[n, n] = -q[n].sum()
    pi = sla.null_space(q.T)[:, 0]
    pi = pi / pi.sum()
    ns = np.arange(size + 1)
    lq = float(np.dot(np.maximum(ns - c, 0), pi))
    return np.array([float(np.dot(ns, pi)), lq, float(pi[0])])


def cases() -> list[Case]:
    out: list[Case] = []
    for c, a in [(1, 0.5), (3, 2.0), (5, 4.2), (10, 7.5)]:
        b = erlang_b(c, a)
        out.append(case(f"erlang-b/{c}/{a}", "erlang-b", [], np.array(b), rtol=1e-12, c=c, a=a))
        out.append(case(f"erlang-c/{c}/{a}", "erlang-c", [], np.array(c * b / (c - a * (1 - b))), rtol=1e-12, c=c, a=a))
    for lam, mu, c in [(2.0, 3.0, 1), (5.0, 2.0, 3), (9.0, 1.0, 12)]:
        m = mmc_measures(lam, mu, c)
        params = {"lam": lam, "mu": mu, "c": c}
        out.append(case(f"mmc/{lam}/{mu}/{c}", "mmc", [], m, rtol=1e-8, atol=1e-12, **params))
    lam = 0.6
    services = {"exponential": (1.0, 1.0), "deterministic": (1.0, 0.0), "uniform": (1.0, 1.0 / 3.0)}
    for name, (mean, var) in services.items():
        rho = lam * mean
        lq = lam**2 * (var + mean**2) / (2 * (1 - rho))
        out.append(case(f"mg1/{name}", "mg1", [], np.array(lq), rtol=1e-12, lam=lam, mean=mean, variance=var))
    return out
