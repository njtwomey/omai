"""Pole placement against scipy.signal.place_poles and python-control's place and acker.

Single-input gains are unique, so they are compared directly; with several inputs the gain depends on the robustness
iterations (and on the bases chosen for the kernels), so the closed-loop poles are compared instead.
"""

from typing import Literal

import control as ct
import numpy as np
import numpy.typing as npt
import scipy
import scipy.signal as sig
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"control": ct.__version__, "scipy": scipy.__version__}


def parts(z: npt.NDArray[np.complex128] | FloatArray) -> FloatArray:
    """Sorted complex values as a [2, n] array of real and imaginary parts."""
    order = sorted(range(len(z)), key=lambda i: (float(np.real(z[i])), float(np.imag(z[i]))))
    return np.array([[float(np.real(z[i])) for i in order], [float(np.imag(z[i])) for i in order]])


def cases() -> list[Case]:
    g = rng(26_2)
    out: list[Case] = []
    a = g.normal(size=(4, 4))
    b1 = g.normal(size=(4, 1))
    real = np.array([-1.0, -2.0, -3.0, -4.0])
    mixed = np.array([-1.0, -2.0, -1.5 + 1.0j, -1.5 - 1.0j])
    for label, poles in {"real": real, "mixed": mixed}.items():
        k = ct.place(a, b1, poles)
        out.append(case(f"single/{label}", "place_gain", [a, b1], k, rtol=1e-7, atol=1e-9, poles=parts(poles)))
        out.append(
            case(f"acker/{label}", "acker", [a, b1], ct.acker(a, b1, poles), rtol=1e-7, atol=1e-9, poles=parts(poles))
        )
    b2 = g.normal(size=(4, 2))
    methods: list[Literal["YT", "KNV0"]] = ["YT", "KNV0"]
    for method in methods:
        for label, poles in {"real": real, "mixed": mixed}.items():
            if method == "KNV0" and label == "mixed":
                continue
            res = sig.place_poles(a, b2, poles, method=method)
            cl = np.linalg.eigvals(a - b2 @ res.gain_matrix)
            out.append(
                case(
                    f"multi/{method}/{label}",
                    "place_poles",
                    [a, b2],
                    parts(cl),
                    rtol=1e-7,
                    atol=1e-8,
                    poles=parts(poles),
                    method=method,
                )
            )
    b4 = g.normal(size=(4, 4))
    res = sig.place_poles(a, b4, mixed)
    out.append(case("full-rank", "place_gain", [a, b4], res.gain_matrix, rtol=1e-8, atol=1e-10, poles=parts(mixed)))
    return out
