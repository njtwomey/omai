"""Baum-Welch against hmmlearn: categorical and Gaussian (diagonal and full) hidden Markov models fitted from fixed
starting parameters for a few iterations, and the log likelihood of the fitted model."""

from importlib.metadata import version

import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"hmmlearn": version("hmmlearn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}

START = np.array([0.6, 0.4])
TRANS = np.array([[0.8, 0.2], [0.3, 0.7]])


def rolls() -> FloatArray:
    g = rng(41)
    states = np.zeros(120, dtype=int)
    for t in range(1, 120):
        states[t] = states[t - 1] if g.random() < 0.9 else 1 - states[t - 1]
    probs = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.3, 0.5]])
    return np.array([g.choice(4, p=probs[s]) for s in states], dtype=np.float64)


def regimes() -> FloatArray:
    g = rng(43)
    states = np.zeros(100, dtype=int)
    for t in range(1, 100):
        states[t] = states[t - 1] if g.random() < 0.9 else 1 - states[t - 1]
    means = np.array([[0.0, 0.0], [3.0, 1.0]])
    return means[states] + g.normal(size=(100, 2))


def cases() -> list[Case]:
    out: list[Case] = []
    x = rolls()
    emission = np.array([[0.3, 0.3, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]])
    m = CategoricalHMM(n_components=2, n_iter=8, tol=-np.inf, init_params="", params="ste")
    m.startprob_ = START
    m.transmat_ = TRANS
    m.emissionprob_ = emission
    m.fit(x[:, None].astype(int))
    output = {
        "startprob": np.asarray(m.startprob_),
        "transmat": np.asarray(m.transmat_),
        "emissionprob": np.asarray(m.emissionprob_),
        "score": np.float64(m.score(x[:, None].astype(int))),
    }
    out.append(case("categorical", "categorical", [x, START, TRANS, emission], output, **TOL))
    y = regimes()
    means = np.array([[0.5, 0.5], [2.0, 0.0]])
    for kind in ["diag", "full"]:
        full = np.array([np.eye(2), np.eye(2) * 1.5])
        g = GaussianHMM(n_components=2, covariance_type=kind, n_iter=8, tol=-np.inf, init_params="", params="stmc")
        g.startprob_ = START
        g.transmat_ = TRANS
        g.means_ = means
        g.covars_ = np.array([np.diag(c) for c in full]) if kind == "diag" else full
        g.fit(y)
        output = {
            "startprob": np.asarray(g.startprob_),
            "transmat": np.asarray(g.transmat_),
            "means": np.asarray(g.means_),
            "covars": np.asarray(g.covars_),
            "score": np.float64(g.score(y)),
        }
        out.append(case(f"gaussian/{kind}", "gaussian", [y, START, TRANS, means, full], output, **TOL, kind=kind))
    return out
