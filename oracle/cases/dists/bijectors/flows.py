"""Flow layers against independent references: the rational-quadratic spline in the form of nflows'
`unconstrained_rational_quadratic_spline` (bins found by `searchsorted`, the tails linear), the LU parameterisation's
log-determinant by scipy.linalg.lu, and the residual-flow log-determinant by numpy's slogdet of I plus a
central-difference Jacobian."""

import numpy as np
import scipy
import scipy.linalg as sla
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def softmax(x: FloatArray) -> FloatArray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return np.asarray(e / e.sum(-1, keepdims=True), dtype=np.float64)


def softplus(x: FloatArray) -> FloatArray:
    return np.asarray(np.logaddexp(x, 0.0), dtype=np.float64)


def knots(unnormalised: FloatArray, min_size: float, bound: float) -> FloatArray:
    k = unnormalised.shape[-1]
    sizes = min_size + (1 - min_size * k) * softmax(unnormalised)
    cumulative = np.cumsum(sizes, axis=-1)
    cumulative = np.concatenate([np.zeros_like(cumulative[..., :1]), cumulative], axis=-1)
    out = 2 * bound * cumulative - bound
    out[..., 0], out[..., -1] = -bound, bound
    return np.asarray(out, dtype=np.float64)


def spline(
    x: FloatArray, uw: FloatArray, uh: FloatArray, ud: FloatArray, inverse: bool, bound: float = 3.0
) -> tuple[FloatArray, FloatArray]:
    """nflows' rational_quadratic_spline with linear tails: returns the outputs and the log |det| of the map applied."""
    cumwidths, cumheights = knots(uw, 1e-3, bound), knots(uh, 1e-3, bound)
    derivatives = np.concatenate([np.ones_like(ud[..., :1]), 1e-3 + softplus(ud), np.ones_like(ud[..., :1])], axis=-1)
    outputs = x.copy()
    logabsdet = np.zeros_like(x)
    for i in range(len(x)):
        if not (-bound <= x[i] <= bound):
            continue
        search = cumheights[i] if inverse else cumwidths[i]
        search = search.copy()
        search[-1] += 1e-6
        b = int(np.searchsorted(search, x[i], side="right") - 1)
        w = cumwidths[i, b + 1] - cumwidths[i, b]
        h = cumheights[i, b + 1] - cumheights[i, b]
        delta = h / w
        d0, d1 = derivatives[i, b], derivatives[i, b + 1]
        if inverse:
            dy = x[i] - cumheights[i, b]
            a = dy * (d0 + d1 - 2 * delta) + h * (delta - d0)
            bb = h * d0 - dy * (d0 + d1 - 2 * delta)
            c = -delta * dy
            root = (2 * c) / (-bb - np.sqrt(bb**2 - 4 * a * c))
            outputs[i] = root * w + cumwidths[i, b]
            theta = root
        else:
            theta = (x[i] - cumwidths[i, b]) / w
            tt = theta * (1 - theta)
            numerator = h * (delta * theta**2 + d0 * tt)
            denominator = delta + (d0 + d1 - 2 * delta) * tt
            outputs[i] = cumheights[i, b] + numerator / denominator
        tt = theta * (1 - theta)
        denominator = delta + (d0 + d1 - 2 * delta) * tt
        derivative_numerator = delta**2 * (d1 * theta**2 + 2 * delta * tt + d0 * (1 - theta) ** 2)
        log_derivative = np.log(derivative_numerator) - 2 * np.log(denominator)
        logabsdet[i] = -log_derivative if inverse else log_derivative
    return outputs, logabsdet


def residual_logdet(x: FloatArray, w: FloatArray) -> FloatArray:
    """log det(I + J) of g(x) = tanh(x W) by central differences, one row at a time."""

    def g(v: FloatArray) -> FloatArray:
        return np.asarray(np.tanh(v @ w), dtype=np.float64)

    out = []
    for row in x:
        h = 1e-6
        j = np.stack([(g(row + h * e) - g(row - h * e)) / (2 * h) for e in np.eye(len(row))], axis=1)
        out.append(np.linalg.slogdet(np.eye(len(row)) + j)[1])
    return np.asarray(out, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(57)
    n, k = 40, 7
    x = np.concatenate([g.uniform(-4.5, 4.5, size=n - 2), np.array([3.0, -3.0])])
    uw, uh, ud = g.normal(size=(n, k)) * 1.5, g.normal(size=(n, k)) * 1.5, g.normal(size=(n, k - 1)) * 1.5
    forward, forward_ld = spline(x, uw, uh, ud, False)
    inverse, inverse_ld = spline(forward, uw, uh, ud, True)
    out: list[Case] = [
        case("spline/forward", "spline.forward", [x, uw, uh, ud], forward, rtol=1e-9, atol=1e-11),
        case("spline/forward-logdet", "spline.forwardLogDet", [x, uw, uh, ud], forward_ld, rtol=1e-9, atol=1e-11),
        case("spline/inverse", "spline.inverse", [forward, uw, uh, ud], inverse, rtol=1e-8, atol=1e-10),
        case("spline/inverse-logdet", "spline.inverseLogDet", [forward, uw, uh, ud], inverse_ld, rtol=1e-8, atol=1e-10),
    ]
    m = g.normal(size=(5, 5)) + 3 * np.eye(5)
    _, _, up = sla.lu(m)
    rows = g.normal(size=(6, 5))
    out.append(case("lu/logdet", "lu.logDet", [m], np.asarray(np.log(np.abs(np.diag(up))).sum())))
    out.append(case("lu/forward", "lu.forward", [m, rows], rows @ m.T, rtol=1e-12))
    w = g.normal(size=(3, 3))
    w = 0.6 * w / np.linalg.norm(w, 2)
    xs = g.normal(size=(4, 3))
    out.append(case("residual/logdet", "residual.logDet", [xs, w], residual_logdet(xs, w), rtol=1e-7, atol=1e-8))
    return out
