"""Discretisation (cont2discrete with every method) against scipy.signal, and Tustin with prewarping and the matched
pole-zero method against python-control's sample_system."""

from typing import Literal

import control as ct
import numpy as np
import scipy
import scipy.signal as sg
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.lti import sorted_parts, stable_system

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "control": ct.__version__}

type Method = Literal["zoh", "foh", "bilinear", "euler", "backward_diff", "gbt", "impulse"]

METHODS: list[tuple[Method, float | None]] = [
    ("zoh", None),
    ("foh", None),
    ("bilinear", None),
    ("euler", None),
    ("backward_diff", None),
    ("gbt", 0.3),
    ("impulse", None),
]


def cases() -> list[Case]:
    g = rng(4163)
    out: list[Case] = []
    a, b, c, _ = stable_system(g, 3, 2, 2)
    d = np.zeros((2, 2))
    d_nonzero = g.normal(size=(2, 2))
    dt = 0.2
    for method, alpha in METHODS:
        dd_in = d if method == "impulse" else d_nonzero
        ad, bd, cd, ddd, _ = sg.cont2discrete((a, b, c, dd_in), dt, method=method, alpha=alpha)
        params: dict[str, float | str] = {"method": method, "dt": dt}
        if alpha is not None:
            params["alpha"] = alpha
        for part, value in zip("abcd", (ad, bd, cd, ddd), strict=True):
            out.append(
                case(
                    f"c2d/ss/{method}/{part}",
                    "c2d-ss",
                    [a, b, c, dd_in],
                    value,
                    rtol=1e-10,
                    atol=1e-12,
                    part=part,
                    **params,
                )
            )
    num, den = [1.0, 0.5], [1.0, 1.2, 2.0]
    tf_methods: list[Method] = ["zoh", "foh", "bilinear", "impulse"]
    for method in tf_methods:
        dnum, dden, _ = sg.cont2discrete((np.array(num), np.array(den)), 0.1, method=method)
        args = [np.array(num), np.array(den)]
        out.append(
            case(
                f"c2d/tf/{method}/num",
                "c2d-tf",
                args,
                np.ravel(dnum),
                rtol=1e-10,
                atol=1e-12,
                method=method,
                dt=0.1,
                part="num",
            )
        )
        out.append(
            case(
                f"c2d/tf/{method}/den", "c2d-tf", args, dden, rtol=1e-10, atol=1e-12, method=method, dt=0.1, part="den"
            )
        )
    z, p, k = [-3.0], [-1.0, -2.0], 4.0
    zd, pd, kd, _ = sg.cont2discrete((z, p, k), 0.25, method="zoh")
    zargs = [np.array(z), np.array(p)]
    out.append(
        case("c2d/zpk/zeros", "c2d-zpk", zargs, sorted_parts(zd), rtol=1e-9, atol=1e-11, gain=k, dt=0.25, part="zeros")
    )
    out.append(
        case("c2d/zpk/poles", "c2d-zpk", zargs, sorted_parts(pd), rtol=1e-9, atol=1e-11, gain=k, dt=0.25, part="poles")
    )
    out.append(case("c2d/zpk/gain", "c2d-zpk", zargs, np.array(kd), rtol=1e-9, gain=k, dt=0.25, part="gain"))

    for w in [1.0, 5.0]:
        sysd = ct.sample_system(ct.ss(a, b, c, d_nonzero), dt, method="bilinear", prewarp_frequency=w)
        for part, value in zip("abcd", (sysd.A, sysd.B, sysd.C, sysd.D), strict=True):
            out.append(
                case(
                    f"c2d/prewarp/{w}/{part}",
                    "c2d-prewarp",
                    [a, b, c, d_nonzero],
                    value,
                    rtol=1e-10,
                    atol=1e-12,
                    dt=dt,
                    w=w,
                    part=part,
                )
            )
    for name, mnum, mden in [("lag", [1.0, 2.0], [1.0, 3.0, 2.5]), ("real", [2.0], [1.0, 3.0, 2.0])]:
        sysd = ct.sample_system(ct.tf(mnum, mden), 0.1, method="matched")
        args = [np.array(mnum), np.array(mden)]
        out.append(
            case(
                f"c2d/matched/{name}/num",
                "c2d-matched",
                args,
                np.atleast_1d(np.squeeze(sysd.num[0][0])),
                rtol=1e-10,
                atol=1e-12,
                dt=0.1,
                part="num",
            )
        )
        out.append(
            case(
                f"c2d/matched/{name}/den",
                "c2d-matched",
                args,
                np.atleast_1d(np.squeeze(sysd.den[0][0])),
                rtol=1e-10,
                atol=1e-12,
                dt=0.1,
                part="den",
            )
        )
    return out
