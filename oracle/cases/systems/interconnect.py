"""Interconnections (series, parallel, feedback, append, connect) against python-control."""

import warnings
from typing import cast

import control as ct
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.lti import feedback, parts, ss, stable_system, tf

REFERENCE: dict[str, str] = {"control": ct.__version__}

W: FloatArray = np.logspace(-1, 1, 5)


def tf_parts(g: ct.TransferFunction) -> tuple[FloatArray, FloatArray]:
    """The (single-channel) numerator and denominator, divided by the leading denominator coefficient."""
    num = np.atleast_1d(np.squeeze(np.asarray(g.num[0][0], dtype=np.float64)))
    den = np.atleast_1d(np.squeeze(np.asarray(g.den[0][0], dtype=np.float64)))
    return num / den[0], den / den[0]


def response(sys: ct.StateSpace | ct.TransferFunction) -> FloatArray:
    """The frequency response at W as `[2, nw * outputs * inputs]` parts, frequency first."""
    h = np.asarray(sys(1j * W))
    if h.ndim == 1:
        return parts(h)
    return parts(np.moveaxis(h, -1, 0).ravel())


def cases() -> list[Case]:
    g = rng(4164)
    out: list[Case] = []
    g1 = tf([1.0, 2.0], [1.0, 3.0, 2.0])
    g2 = tf([3.0], [1.0, 0.5])
    tf_in = [np.array([1.0, 2.0]), np.array([1.0, 3.0, 2.0]), np.array([3.0]), np.array([1.0, 0.5])]
    combos: list[tuple[str, ct.TransferFunction, dict[str, float | str]]] = [
        ("series", cast(ct.TransferFunction, ct.series(g1, g2)), {"op": "series"}),
        ("parallel", cast(ct.TransferFunction, ct.parallel(g1, g2)), {"op": "parallel"}),
        ("feedback", cast(ct.TransferFunction, feedback(g1, g2)), {"op": "feedback", "sign": -1}),
        ("feedback-positive", cast(ct.TransferFunction, feedback(g1, g2, 1)), {"op": "feedback", "sign": 1}),
        ("unity", cast(ct.TransferFunction, feedback(g1, 1)), {"op": "unity", "sign": -1}),
    ]
    for name, result, params in combos:
        num, den = tf_parts(result)
        out.append(case(f"tf/{name}/num", "tf-op", tf_in, num, rtol=1e-12, atol=1e-13, part="num", **params))
        out.append(case(f"tf/{name}/den", "tf-op", tf_in, den, rtol=1e-12, atol=1e-13, part="den", **params))

    a1, b1, c1, d1 = stable_system(g, 3, 2, 2)
    a2, b2, c2, d2 = stable_system(g, 2, 2, 2)
    s1 = ss(a1, b1, c1, d1)
    s2 = ss(a2, b2, c2, 0.3 * d2)
    ss_in = [a1, b1, c1, d1, a2, b2, c2, 0.3 * d2]
    results: list[tuple[str, ct.StateSpace | ct.TransferFunction]] = [
        ("series", cast(ct.StateSpace, ct.series(s1, s2))),
        ("parallel", cast(ct.StateSpace, ct.parallel(s1, s2))),
        ("feedback", feedback(s1, s2)),
        ("feedback-positive", feedback(s1, s2, 1)),
    ]
    for name, result in results:
        sign = 1 if name.endswith("positive") else -1
        op = name.split("-")[0]
        out.append(case(f"ss/{name}", "ss-op", ss_in, response(result), rtol=1e-9, atol=1e-11, op=op, sign=sign))

    # append then connect: the second block is driven by the first, which gets negative feedback from the second.
    blocks = ct.append(ct.tf2ss(g1), ct.tf2ss(g2))
    q = [[2, 1], [1, -2]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = cast(ct.StateSpace, ct.connect(blocks, q, [1], [1, 2]))
    out.append(case("connect", "connect", tf_in, response(loop), rtol=1e-9, atol=1e-11))
    return out
