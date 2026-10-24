"""Second-order sections (zpk2sos with every pairing, tf2sos, sos2tf, sos2zpk) against scipy.signal."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy
import scipy.signal as sg
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.lti import parts, sorted_parts

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type ComplexArray = npt.NDArray[np.complex128]


def zpk_of(design: tuple[object, object, object]) -> tuple[ComplexArray, ComplexArray, float]:
    """A filter design's zeros, poles and gain with fixed dtypes."""
    z, p, k = design
    return np.asarray(z, dtype=np.complex128), np.asarray(p, dtype=np.complex128), float(np.real(np.asarray(k)))


def cases() -> list[Case]:
    out: list[Case] = []
    digital: list[tuple[str, ComplexArray, ComplexArray, float]] = [
        ("butter4", *zpk_of(sg.butter(4, 0.3, output="zpk"))),
        ("ellip5", *zpk_of(sg.ellip(5, 0.5, 40, 0.2, output="zpk"))),
        ("cheby1-3", *zpk_of(sg.cheby1(3, 1, 0.4, output="zpk"))),
        (
            "docs",
            np.array([-1, -0.5 - 0.5j, -0.5 + 0.5j]),
            np.array([0.75, 0.8 + 0.1j, 0.8 - 0.1j], dtype=np.complex128),
            1.0,
        ),
        ("real", np.array([0.2, -0.4], dtype=np.complex128), np.array([0.5, -0.3, 0.9], dtype=np.complex128), 2.0),
    ]
    for name, z, p, k in digital:
        pairings: list[Literal["nearest", "keep_odd", "minimal"]] = ["nearest", "keep_odd", "minimal"]
        for pairing in pairings:
            sos = sg.zpk2sos(z, p, k, pairing=pairing)
            out.append(
                case(
                    f"zpk2sos/{name}/{pairing}",
                    "zpk2sos",
                    [parts(z), parts(p)],
                    sos,
                    rtol=1e-10,
                    atol=1e-12,
                    gain=k,
                    pairing=pairing,
                    analog=False,
                )
            )
        sos = sg.zpk2sos(z, p, k)
        b, a = sg.sos2tf(sos)
        out.append(case(f"sos2tf/{name}/num", "sos2tf", [sos], b, rtol=1e-12, atol=1e-14, part="num"))
        out.append(case(f"sos2tf/{name}/den", "sos2tf", [sos], a, rtol=1e-12, atol=1e-14, part="den"))
        sz, sp, sk = sg.sos2zpk(sos)
        out.append(
            case(f"sos2zpk/{name}/zeros", "sos2zpk", [sos], sorted_parts(sz), rtol=1e-8, atol=1e-9, part="zeros")
        )
        out.append(
            case(f"sos2zpk/{name}/poles", "sos2zpk", [sos], sorted_parts(sp), rtol=1e-9, atol=1e-10, part="poles")
        )
        out.append(case(f"sos2zpk/{name}/gain", "sos2zpk", [sos], np.array(sk), rtol=1e-12, part="gain"))
        bb, aa = sg.zpk2tf(z, p, k)
        # Repeated zeros at -1 (Butterworth, Chebyshev I) come back from either library's root finder with only about
        # a fourth of the digits (an m-fold root perturbs as eps^(1/m)), so those sections agree to that level.
        repeated = name in ("butter4", "cheby1-3")
        out.append(
            case(
                f"tf2sos/{name}",
                "tf2sos",
                [np.real(bb), np.real(aa)],
                sg.tf2sos(np.real(bb), np.real(aa)),
                rtol=1e-3 if repeated else 1e-8,
                atol=1e-4 if repeated else 1e-10,
            )
        )
    analog: list[tuple[str, ComplexArray, ComplexArray, float]] = [
        ("butter4", *zpk_of(sg.butter(4, 2.0, analog=True, output="zpk"))),
        ("cheby2-5", *zpk_of(sg.cheby2(5, 30, 1.5, analog=True, output="zpk"))),
    ]
    for name, z, p, k in analog:
        out.append(
            case(
                f"zpk2sos/analog/{name}",
                "zpk2sos",
                [parts(z), parts(p)],
                sg.zpk2sos(z, p, k, analog=True),
                rtol=1e-10,
                atol=1e-12,
                gain=k,
                pairing="minimal",
                analog=True,
            )
        )
    return out
