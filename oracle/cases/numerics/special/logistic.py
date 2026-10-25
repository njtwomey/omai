"""Logistic and entropy helpers against numpy and scipy."""

import numpy as np
import scipy
import scipy.special as sp
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, reference, rng, softplus

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def log1mexp(x: FloatArray) -> FloatArray:
    # The standard stable form (Machler 2012): log(-expm1(x)) near 0, log1p(-exp(x)) further out.
    return np.where(x > -np.log(2), np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def cases() -> list[Case]:
    g = rng(3)
    x = np.concatenate([g.uniform(-30, 30, 30), [0.0, -700.0, 700.0]])
    p = g.uniform(0.001, 0.999, 30)
    neg = -g.uniform(0.001, 20, 30)
    a, b = g.uniform(-20, 20, 30), g.uniform(-20, 20, 30)
    u, v = g.uniform(0.01, 10, 30), g.uniform(0.01, 10, 30)
    edges_x = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 2.0])
    edges_y = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    return [
        reference("expit", "expit", sp.expit, [x]),
        reference("logit", "logit", sp.logit, [p]),
        reference("logSigmoid", "logSigmoid", sp.log_expit, [x]),
        reference("softplus", "softplus", softplus, [x]),
        reference("log1mexp", "log1mexp", log1mexp, [neg]),
        reference("logaddexp", "logaddexp", np.logaddexp, [a, b]),
        reference("xlogy", "xlogy", sp.xlogy, [a, u]),
        reference("xlog1py", "xlog1py", sp.xlog1py, [a, u]),
        reference("entr", "entr", sp.entr, [np.concatenate([u, [0.0, -1.0]])]),
        reference("relEntr", "relEntr", sp.rel_entr, [u, v]),
        reference("klDiv", "klDiv", sp.kl_div, [u, v]),
        case("xlogy/edges", "xlogy", [edges_x, np.abs(edges_y)], sp.xlogy(edges_x, np.abs(edges_y))),
        case("relEntr/edges", "relEntr", [edges_x, edges_y], sp.rel_entr(edges_x, edges_y)),
    ]
