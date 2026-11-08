"""Inverse Laplace and z-transforms by partial fractions against scipy.signal's impulse responses (impulse, lfilter),
and difference equations against freqz."""

import numpy as np
import scipy
import scipy.signal as sg
from omai_oracle.fixture import Case, case
from omai_oracle.lti import parts

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

LAPLACE: list[tuple[str, list[float], list[float]]] = [
    ("distinct", [1.0, 2.0], [1.0, 6.0, 11.0, 6.0]),
    ("complex", [2.0, 1.0], [1.0, 0.4, 4.04]),
    ("double", [1.0, 0.0], [1.0, 4.0, 4.0]),
    ("triple", [3.0], [1.0, 3.0, 3.0, 1.0]),
]

Z: list[tuple[str, list[float], list[float]]] = [
    ("first-order", [1.0, 0.5], [1.0, -0.6]),
    ("resonant", [0.2, 0.1, 0.05], [1.0, -1.2, 0.72]),
    ("double", [1.0], [1.0, -1.0, 0.25]),
    ("fir-and-pole", [1.0, 0.3, -0.2, 0.1], [1.0, -0.5]),
]


def cases() -> list[Case]:
    out: list[Case] = []
    t = np.linspace(0.0, 4.0, 17)
    for name, num, den in LAPLACE:
        _, h = sg.impulse((num, den), T=t)
        out.append(
            case(
                f"inverseLaplace/{name}", "inverseLaplace", [np.array(num), np.array(den), t], h, rtol=1e-8, atol=1e-10
            )
        )
    impulse = np.zeros(12)
    impulse[0] = 1.0
    w = np.linspace(0.1, 3.0, 6)
    for name, b, a in Z:
        out.append(
            case(
                f"inverseZ/{name}",
                "inverseZ",
                [np.array(b), np.array(a)],
                sg.lfilter(b, a, impulse),
                rtol=1e-8,
                atol=1e-10,
                n=12,
            )
        )
        _, hz = sg.freqz(b, a, worN=w)
        out.append(
            case(
                f"differenceToTf/{name}",
                "differenceToTf",
                [np.array(b), np.array(a), w],
                parts(hz),
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
