"""Reductions and scans against numpy and scipy."""

from typing import Any

import numpy as np
import scipy.special as sp
from whitebox_oracle.fixture import case
from whitebox_oracle.helpers import rng, uniform

REFERENCE = {"numpy": np.__version__}

AXES = [None, 0, -1, (0, 2)]


def cases():
    x = uniform(rng(400), (2, 3, 4))
    out = []
    for axis in AXES:
        label = "all" if axis is None else str(axis)
        params: dict[str, Any] = {} if axis is None else {"axis": list(axis) if isinstance(axis, tuple) else axis}
        out.append(case(f"sum/{label}", "sum", [x], np.sum(x, axis=axis), **params))
        out.append(case(f"prod/{label}", "prod", [x], np.prod(x, axis=axis), **params))
        out.append(case(f"mean/{label}", "mean", [x], np.mean(x, axis=axis), **params))
        out.append(case(f"max/{label}", "max", [x], np.max(x, axis=axis), **params))
        out.append(case(f"min/{label}", "min", [x], np.min(x, axis=axis), **params))
        out.append(case(f"variance/{label}", "variance", [x], np.var(x, axis=axis), **params))
        out.append(case(f"variance-ddof1/{label}", "variance", [x], np.var(x, axis=axis, ddof=1), ddof=1, **params))
        out.append(case(f"std/{label}", "std", [x], np.std(x, axis=axis), **params))
        out.append(case(f"logsumexp/{label}", "logsumexp", [x], sp.logsumexp(x, axis=axis), **params))
    out.append(case("sum/keepDims", "sum", [x], np.sum(x, axis=1, keepdims=True), axis=1, keepDims=True))
    for axis in [0, 1, -1]:
        out.append(case(f"argmax/{axis}", "argmax", [x], np.argmax(x, axis=axis).astype(np.int32), axis=axis))
        out.append(case(f"argmin/{axis}", "argmin", [x], np.argmin(x, axis=axis).astype(np.int32), axis=axis))
        out.append(case(f"cumsum/{axis}", "cumsum", [x], np.cumsum(x, axis=axis), axis=axis))
        out.append(case(f"cumprod/{axis}", "cumprod", [x], np.cumprod(x, axis=axis), axis=axis))
        out.append(case(f"softmax/{axis}", "softmax", [x], sp.softmax(x, axis=axis), axis=axis))
        out.append(case(f"logSoftmax/{axis}", "logSoftmax", [x], sp.log_softmax(x, axis=axis), axis=axis))
    big = np.array([[1000.0, 1000.0, -1000.0], [-np.inf, -np.inf, -np.inf]])
    out.append(case("logsumexp/extremes", "logsumexp", [big], sp.logsumexp(big, axis=1), axis=1))
    v = uniform(rng(401), (6,))
    for ord_ in [1, 2, 3, np.inf, -np.inf]:
        out.append(case(f"norm/{ord_}", "norm", [v], np.linalg.norm(v, ord=ord_), ord=float(ord_)))
    return out
