"""Minimal realisations against python-control (minreal for state-space and transfer-function systems)."""

from typing import cast

import control as ct
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from omai_oracle.lti import parts

REFERENCE: dict[str, str] = {"control": ct.__version__}


def arr(x: object) -> FloatArray:
    """A python-control matrix as a float array."""
    return np.asarray(x, dtype=np.float64)


def ss_cases() -> list[tuple[str, FloatArray, FloatArray, FloatArray, FloatArray]]:
    """State-space systems with uncontrollable or unobservable parts."""
    redundant = cast(ct.StateSpace, ct.tf2ss(ct.tf([1.0, 1.0], [1.0, 3.0, 2.0])))
    t = np.array([[1.0, 0.5, 0.2], [0.0, 1.0, -0.3], [0.4, 0.0, 1.0]])
    a = np.diag([-1.0, -2.0, -3.0])
    b = np.array([[1.0], [0.0], [1.0]])
    c = np.array([[1.0, 1.0, 0.0]])
    mimo_a = np.diag([-1.0, -2.0, -0.5, -4.0])
    mimo_b = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    mimo_c = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])
    return [
        ("cancelling", arr(redundant.A), arr(redundant.B), arr(redundant.C), arr(redundant.D)),
        ("modal", a, b, c, np.zeros((1, 1))),
        ("hidden", np.linalg.solve(t, a @ t), np.linalg.solve(t, b), c @ t, np.array([[0.5]])),
        ("mimo", mimo_a, mimo_b, mimo_c, np.zeros((2, 2))),
        (
            "minimal",
            np.array([[-1.0, 1.0], [-2.0, -0.5]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0, 0.0]]),
            np.zeros((1, 1)),
        ),
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, a, b, c, d in ss_cases():
        reduced = cast(ct.StateSpace, ct.minreal(ct.ss(a, b, c, d), verbose=False))
        out.append(case(f"minreal/ss/{name}/order", "minreal-ss-order", [a, b, c, d], np.array(reduced.nstates)))
        # A minimal realisation is unique only up to similarity: compare invariants. Multivariable systems are compared
        # by their frequency response (each channel of a minimal MIMO system need not be minimal on its own).
        if reduced.ninputs > 1 or reduced.noutputs > 1:
            w = np.logspace(-1, 1, 5)
            resp = np.moveaxis(reduced(1j * w), -1, 0).ravel()
            out.append(
                case(
                    f"minreal/ss/{name}/response",
                    "minreal-ss-freq",
                    [a, b, c, d, w],
                    parts(resp),
                    rtol=1e-9,
                    atol=1e-11,
                )
            )
            continue
        for i in range(reduced.ninputs):
            for o in range(reduced.noutputs):
                g = cast(ct.TransferFunction, ct.tf(reduced[o, i]))
                num = np.atleast_1d(np.squeeze(g.num[0][0]))
                den = np.atleast_1d(np.squeeze(g.den[0][0]))
                num = np.concatenate([np.zeros(len(den) - len(num)), num]) / den[0]
                out.append(
                    case(
                        f"minreal/ss/{name}/{o}{i}/num",
                        "minreal-ss-tf",
                        [a, b, c, d],
                        num,
                        rtol=1e-8,
                        atol=1e-10,
                        input=i,
                        output_index=o,
                        part="num",
                    )
                )
                out.append(
                    case(
                        f"minreal/ss/{name}/{o}{i}/den",
                        "minreal-ss-tf",
                        [a, b, c, d],
                        den / den[0],
                        rtol=1e-8,
                        atol=1e-10,
                        input=i,
                        output_index=o,
                        part="den",
                    )
                )
    tfs: list[tuple[str, list[float], list[float]]] = [
        ("first", [1.0, 1.0], [1.0, 3.0, 2.0]),
        ("double", [1.0, 3.0, 2.0], [1.0, 4.0, 5.0, 2.0]),
        ("complex", [1.0, 0.4, 1.04], [1.0, 1.4, 1.44, 1.04]),
        ("none", [1.0, 2.0], [1.0, 3.0, 5.0]),
    ]
    for name, num, den in tfs:
        reduced = ct.minreal(ct.tf(num, den), verbose=False)
        rnum = np.atleast_1d(np.squeeze(reduced.num[0][0]))
        rden = np.atleast_1d(np.squeeze(reduced.den[0][0]))
        args = [np.array(num), np.array(den)]
        out.append(case(f"minreal/tf/{name}/num", "minreal-tf", args, rnum, rtol=1e-7, atol=1e-9, part="num"))
        out.append(case(f"minreal/tf/{name}/den", "minreal-tf", args, rden, rtol=1e-7, atol=1e-9, part="den"))
    return out
