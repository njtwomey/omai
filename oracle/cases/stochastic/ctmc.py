"""Continuous-time Markov chains against scipy: transition matrices exp(tQ) (scipy.linalg.expm), stationary laws (the
null space of Q^T by scipy.linalg.null_space), expected hitting times (numpy.linalg.solve of -Q_UU k = 1), the
embedded jump chain and uniformisation."""

import numpy as np
import scipy
import scipy.linalg as sla
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def generator(g: np.random.Generator, n: int) -> FloatArray:
    q = g.uniform(0.1, 2.0, size=(n, n))
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def cases() -> list[Case]:
    g = rng(5150)
    out: list[Case] = []
    for n in [2, 4]:
        q = generator(g, n)
        for t in [0.1, 1.3]:
            out.append(case(f"transition/{n}/{t}", "transition", [q], sla.expm(q * t), rtol=1e-10, atol=1e-13, t=t))
        ns = sla.null_space(q.T)[:, 0]
        out.append(case(f"stationary/{n}", "stationary", [q], ns / ns.sum(), rtol=1e-10, atol=1e-13))
        rest = list(range(1, n))
        k = np.zeros(n)
        k[rest] = np.linalg.solve(-q[np.ix_(rest, rest)], np.ones(n - 1))
        out.append(case(f"hitting/{n}", "hitting", [q], k, rtol=1e-10, target=[0]))
        rates = -np.diag(q)
        jump = q / rates[:, None]
        np.fill_diagonal(jump, 0)
        out.append(case(f"embedded/{n}", "embedded", [q], jump, rtol=1e-12))
        lam = float(rates.max()) * 1.5
        out.append(case(f"uniformise/{n}", "uniformise", [q], np.eye(n) + q / lam, rtol=1e-12, rate=lam))
    return out
