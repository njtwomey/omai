"""Branching processes: Galton-Watson extinction probabilities (scipy.optimize.brentq on G(s) - s below the minimiser
of scipy.optimize.minimize_scalar), extinction by generation (iterated numpy polyval), birth-death stationary laws
(scipy.linalg.null_space of the generator), the linear birth-death extinction probability by time t (scipy.linalg.expm
of the truncated generator) and the Yule process law (scipy.stats.geom)."""

import numpy as np
import scipy
import scipy.linalg as sla
import scipy.optimize as so
import scipy.stats as st
from numpy.polynomial import polynomial as npoly
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def extinction(pmf: FloatArray) -> float:
    def f(s: float) -> float:
        return float(npoly.polyval(s, pmf)) - s

    mean = float(np.dot(np.arange(len(pmf)), pmf))
    if mean <= 1:
        return 1.0
    lowest = so.minimize_scalar(f, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
    return float(so.brentq(f, 0.0, float(lowest.x), xtol=1e-15, rtol=1e-15))


def by_generation(pmf: FloatArray, n: int) -> FloatArray:
    out = [0.0]
    for _ in range(n):
        out.append(float(npoly.polyval(out[-1], pmf)))
    return np.array(out)


def generator(birth: FloatArray, death: FloatArray) -> FloatArray:
    n = len(birth)
    q = np.zeros((n, n))
    for i in range(n):
        if i + 1 < n:
            q[i, i + 1] = birth[i]
        if i > 0:
            q[i, i - 1] = death[i]
        q[i, i] = -q[i].sum()
    return q


def cases() -> list[Case]:
    out: list[Case] = []
    laws = [
        np.array([0.25, 0.25, 0.5]),
        np.array([0.2, 0.3, 0.3, 0.2]),
        np.array([0.1, 0.0, 0.0, 0.9]),
        np.array([0.5, 0.5]),
    ]
    for k, pmf in enumerate(laws):
        out.append(case(f"extinction/{k}", "extinction", [pmf], np.array(extinction(pmf)), rtol=1e-12, atol=1e-14))
        out.append(case(f"by-generation/{k}", "by-generation", [pmf], by_generation(pmf, 6), rtol=1e-12, steps=6))
    # birth-death with lambda_n = 2 / (n + 1), mu_n = n (a discouraged-arrival queue) on {0, ..., 12}
    n = 12
    birth = np.array([2 / (i + 1) for i in range(n + 1)])
    birth[n] = 0
    death = np.arange(n + 1, dtype=np.float64)
    ns = sla.null_space(generator(birth, death).T)[:, 0]
    out.append(case("birth-death/stationary", "birth-death-stationary", [], ns / ns.sum(), rtol=1e-10, n=n))
    for lam, mu in [(1.0, 1.5), (2.0, 1.0), (1.2, 1.2)]:
        # Truncation at 400 individuals leaves P(X_t = 0) exact to rounding for these horizons.
        size = 400
        b = np.array([lam * i for i in range(size + 1)])
        b[size] = 0
        d = np.array([mu * i for i in range(size + 1)])
        q = generator(b, d)
        times = [0.5, 2.0]
        values = np.array([sla.expm(q * t)[1, 0] for t in times])
        out.append(
            case(f"linear-bd/{lam}/{mu}", "linear-bd", [np.array(times)], values, rtol=1e-8, atol=1e-12, lam=lam, mu=mu)
        )
    ks = np.array([1.0, 2.0, 5.0, 9.0])
    for lam, t in [(1.0, 0.7), (0.4, 3.0)]:
        out.append(case(f"yule/{lam}/{t}", "yule", [ks], st.geom.pmf(ks, np.exp(-lam * t)), rtol=1e-12, lam=lam, t=t))
    return out
