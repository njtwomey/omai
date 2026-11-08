"""Normal-distribution functions against scipy."""

import numpy as np
import scipy
import scipy.special as sp
from omai_oracle.fixture import Case
from omai_oracle.helpers import FloatArray, reference, rng
from scipy.stats import norm

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def log_interval(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.log(sp.ndtr(b) - sp.ndtr(a))


def cases() -> list[Case]:
    g = rng(2)
    x = np.concatenate([g.uniform(-6, 6, 30), [0.0, -20.0, 10.0]])
    p = np.concatenate([g.uniform(0.001, 0.999, 30), [0.5, 1e-10, 0.0, 1.0]])
    h, a = g.uniform(-3, 3, 20), g.uniform(-3, 3, 20)
    lo = g.uniform(-3, 2, 20)
    hi = lo + g.uniform(0.1, 3, 20)
    return [
        reference("ndtr", "ndtr", sp.ndtr, [x]),
        reference("logNdtr", "logNdtr", sp.log_ndtr, [x]),
        reference("ndtri", "ndtri", sp.ndtri, [p]),
        reference("normalPdf", "normalPdf", norm.pdf, [x]),
        reference("normalLogPdf", "normalLogPdf", norm.logpdf, [x]),
        reference("owensT", "owensT", sp.owens_t, [h, a]),
        reference("logIntervalProbability", "logIntervalProbability", log_interval, [lo, hi]),
    ]
