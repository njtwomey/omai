"""LTI conversions (tf2ss, ss2tf, tf2zpk, zpk2tf, zpk2ss, ss2zpk, normalize) against scipy.signal."""

import warnings

import numpy as np
import numpy.typing as npt
import scipy
import scipy.signal as sg
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.lti import parts, sorted_parts, stable_system

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type ComplexArray = npt.NDArray[np.complex128]

TFS: list[tuple[str, list[float], list[float]]] = [
    ("second-order", [1.0], [1.0, 3.0, 2.0]),
    ("proper", [2.0, -1.0, 0.5], [1.0, 0.4, 2.0]),
    ("unnormalised", [0.0, 3.0, 6.0], [2.0, 1.0, 4.0, 2.0]),
    ("leading-zeros", [0.0, 0.0, 1.0, 2.0], [0.0, 4.0, 2.0, 1.0, 0.5]),
    ("complex-poles", [1.0, 0.0, -2.0], [1.0, 0.2, 1.0, 0.1]),
    ("gain", [3.0], [2.0]),
]


def cases() -> list[Case]:
    g = rng(4160)
    out: list[Case] = []
    warnings.simplefilter("ignore", sg.BadCoefficients)
    for name, num, den in TFS:
        b, a = sg.normalize(num, den)
        out.append(case(f"normalize/{name}", "normalize-num", [np.array(num), np.array(den)], b, rtol=1e-13))
        out.append(case(f"normalize/{name}/den", "normalize-den", [np.array(num), np.array(den)], a, rtol=1e-13))
        realisation = sg.tf2ss(num, den)
        for part, value in zip("abcd", realisation, strict=True):
            out.append(case(f"tf2ss/{name}/{part}", "tf2ss", [np.array(num), np.array(den)], value, part=part))
        if len(sg.normalize(num, den)[0]) > 0 and np.any(np.asarray(num) != 0):
            z, p, k = sg.tf2zpk(num, den)
            out.append(
                case(
                    f"tf2zpk/{name}/zeros",
                    "tf2zpk",
                    [np.array(num), np.array(den)],
                    sorted_parts(z),
                    rtol=1e-10,
                    atol=1e-12,
                    part="zeros",
                )
            )
            out.append(
                case(
                    f"tf2zpk/{name}/poles",
                    "tf2zpk",
                    [np.array(num), np.array(den)],
                    sorted_parts(p),
                    rtol=1e-10,
                    atol=1e-12,
                    part="poles",
                )
            )
            out.append(
                case(
                    f"tf2zpk/{name}/gain",
                    "tf2zpk",
                    [np.array(num), np.array(den)],
                    np.array(k),
                    rtol=1e-13,
                    part="gain",
                )
            )
    simo_num = np.array([[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]])
    simo_den = np.array([1.0, 0.5, 3.0])
    realisation = sg.tf2ss(simo_num, simo_den)
    for part, value in zip("abcd", realisation, strict=True):
        out.append(case(f"tf2ss/simo/{part}", "tf2ss", [simo_num, simo_den], value, part=part))

    for n, m, p in [(1, 1, 1), (3, 1, 1), (3, 2, 2), (4, 2, 3)]:
        a, b, c, d = stable_system(g, n, m, p)
        for inp in range(m):
            num, den = sg.ss2tf(a, b, c, d, input=inp)
            label = f"ss2tf/{n}x{m}x{p}/input{inp}"
            out.append(case(f"{label}/num", "ss2tf", [a, b, c, d], num, rtol=1e-10, atol=1e-11, part="num", input=inp))
            out.append(case(f"{label}/den", "ss2tf", [a, b, c, d], den, rtol=1e-10, atol=1e-11, part="den", input=inp))
        z, pp, k = sg.ss2zpk(a, b, c[:1], d[:1], input=0)
        args = [a, b, c[:1], d[:1]]
        out.append(case(f"ss2zpk/{n}/zeros", "ss2zpk", args, sorted_parts(z), rtol=1e-8, atol=1e-9, part="zeros"))
        out.append(case(f"ss2zpk/{n}/poles", "ss2zpk", args, sorted_parts(pp), rtol=1e-9, atol=1e-10, part="poles"))
        out.append(case(f"ss2zpk/{n}/gain", "ss2zpk", args, np.array(k), rtol=1e-10, atol=1e-12, part="gain"))

    bz, bp, bk = sg.butter(4, 0.3, output="zpk")
    zpks: list[tuple[str, ComplexArray, ComplexArray, float]] = [
        ("real", np.array([-1.0, 2.0], dtype=np.complex128), np.array([-0.5, -2.0, -3.0], dtype=np.complex128), 2.5),
        ("pairs", np.array([-1 + 2j, -1 - 2j]), np.array([-0.3 + 1j, -0.3 - 1j, -4.0]), 0.7),
        ("butter", np.asarray(bz, dtype=np.complex128), np.asarray(bp, dtype=np.complex128), float(bk)),
    ]
    for name, z, pp, k in zpks:
        num, den = sg.zpk2tf(z, pp, k)
        out.append(
            case(
                f"zpk2tf/{name}/num",
                "zpk2tf",
                [parts(z), parts(pp)],
                np.real(num),
                rtol=1e-12,
                atol=1e-14,
                gain=float(k),
                part="num",
            )
        )
        out.append(
            case(
                f"zpk2tf/{name}/den",
                "zpk2tf",
                [parts(z), parts(pp)],
                np.real(den),
                rtol=1e-12,
                atol=1e-14,
                gain=float(k),
                part="den",
            )
        )
        realisation = sg.zpk2ss(z, pp, k)
        for part, value in zip("abcd", realisation, strict=True):
            out.append(
                case(
                    f"zpk2ss/{name}/{part}",
                    "zpk2ss",
                    [parts(z), parts(pp)],
                    np.real(value),
                    rtol=1e-12,
                    atol=1e-14,
                    gain=float(k),
                    part=part,
                )
            )
    return out
