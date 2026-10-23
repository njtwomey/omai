"""Logistic and entropy helpers against mpmath."""

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def expit(v: Mpf) -> Mpf:
    return 1 / (1 + mp.exp(-v))


def logit(v: Mpf) -> Mpf:
    return mp.log(v / (1 - v))


def log_sigmoid(v: Mpf) -> Mpf:
    return -mp.log(1 + mp.exp(-v))


def softplus(v: Mpf) -> Mpf:
    return mp.log(1 + mp.exp(v))


def log1mexp(v: Mpf) -> Mpf:
    return mp.log(1 - mp.exp(v))


def logaddexp(u: Mpf, v: Mpf) -> Mpf:
    return mp.log(mp.exp(u) + mp.exp(v))


def xlogy(u: Mpf, v: Mpf) -> Mpf:
    return u * mp.log(v)


def xlog1py(u: Mpf, v: Mpf) -> Mpf:
    return u * mp.log1p(v)


def entr(v: Mpf) -> Mpf:
    return -v * mp.log(v)


def rel_entr(u: Mpf, v: Mpf) -> Mpf:
    return u * mp.log(u / v)


def kl_div(u: Mpf, v: Mpf) -> Mpf:
    return u * mp.log(u / v) - u + v


def cases() -> list[Case]:
    g = rng(3)
    x = np.concatenate([g.uniform(-30, 30, 30), [-700.0, 700.0]])
    p = g.uniform(1e-6, 1 - 1e-6, 30)
    neg = -np.exp(g.uniform(np.log(1e-12), np.log(50), 30))
    a, b = g.uniform(-50, 50, 30), g.uniform(-50, 50, 30)
    pos1, pos2 = np.exp(g.uniform(-5, 5, 30)), np.exp(g.uniform(-5, 5, 30))
    out: list[Case] = []
    out += value_and_grads("expit", "expit", expit, [x], wrt=[0], atol=1e-300, dps=400)
    out += value_and_grads("logit", "logit", logit, [p], wrt=[0])
    out += value_and_grads("logSigmoid", "logSigmoid", log_sigmoid, [x], wrt=[0], dps=400)
    out += value_and_grads("softplus", "softplus", softplus, [x], wrt=[0], atol=1e-300, dps=400)
    out += value_and_grads("log1mexp", "log1mexp", log1mexp, [neg], wrt=[0])
    out += value_and_grads("logaddexp", "logaddexp", logaddexp, [a, b], wrt=[0, 1])
    out += value_and_grads("xlogy", "xlogy", xlogy, [a, pos1], wrt=[0, 1])
    out += value_and_grads("xlog1py", "xlog1py", xlog1py, [a, pos1], wrt=[0, 1])
    out += value_and_grads("entr", "entr", entr, [pos1], wrt=[0])
    out += value_and_grads("relEntr", "relEntr", rel_entr, [pos1, pos2], wrt=[0, 1])
    out += value_and_grads("klDiv", "klDiv", kl_div, [pos1, pos2], wrt=[0, 1], atol=1e-14)
    edges_x = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 2.0])
    edges_y = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    out.append(case("xlogy/edges", "xlogy", [edges_x, np.abs(edges_y)], np.array([0, 0, 0, -np.inf, 0, 0.0])))
    out.append(case("relEntr/edges", "relEntr", [edges_x, edges_y], np.array([0, 0, np.inf, np.inf, np.inf, np.inf])))
    out.append(case("entr/edges", "entr", [np.array([0.0, -1.0])], np.array([0.0, -np.inf])))
    return out
