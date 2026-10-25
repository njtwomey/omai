"""Discrete wavelet transforms against PyWavelets: filter banks, dwt and idwt in every mode, wavedec and waverec,
the 2-D transforms."""

from typing import Any, Literal

import numpy as np
import pywt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from pywt._extensions._pywt import Wavelet  # pywt re-exports it untyped

REFERENCE: dict[str, str] = {"pywt": pywt.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}

type Mode = Literal[
    "zero", "constant", "symmetric", "reflect", "periodic", "smooth", "antisymmetric", "antireflect", "periodization"
]
MODES: list[Mode] = [
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
    "periodization",
]
WAVELETS = ["haar", "db2", "db7", "sym5", "coif3", "bior3.5", "rbio2.4", "dmey"]


def cases() -> list[Case]:
    g = rng(412)
    out: list[Case] = []
    for name in WAVELETS:
        bank = np.array(Wavelet(name).filter_bank)
        out.append(case(f"filters/{name}", "filters", [], bank, rtol=1e-14, atol=1e-16, wavelet=name))
    x = g.normal(size=19)
    for name in ["db2", "sym5", "bior3.5"]:
        for mode in MODES:
            ca, cd = pywt.dwt(x, name, mode=mode)
            params: dict[str, Any] = {"wavelet": name, "mode": mode}
            out.append(case(f"dwt/{name}/{mode}", "dwt", [x], np.stack([ca, cd]), **TOL, **params))
            out.append(
                case(f"idwt/{name}/{mode}", "idwt", [ca, cd], pywt.idwt(ca, cd, name, mode=mode), **TOL, **params)
            )
    m = g.normal(size=(3, 12))
    ca, cd = pywt.dwt(m, "db3", axis=0)
    out.append(case("dwt/axis0", "dwt", [m], np.stack([ca, cd]), **TOL, wavelet="db3", axis=0))
    y = g.normal(size=50)
    for name, level, mode in [("db4", None, "symmetric"), ("haar", 3, "periodization"), ("coif2", 2, "reflect")]:
        coeffs = pywt.wavedec(y, name, level=level, mode=mode)
        flat = np.concatenate(coeffs)
        params2: dict[str, Any] = {"wavelet": name, "mode": mode, **({} if level is None else {"level": level})}
        out.append(case(f"wavedec/{name}/{level}/{mode}", "wavedec", [y], flat, **TOL, **params2))
        out.append(
            case(
                f"waverec/{name}/{level}/{mode}",
                "waverec",
                [y],
                pywt.waverec(coeffs, name, mode=mode),
                **TOL,
                **params2,
            )
        )
    img = g.normal(size=(9, 12))
    for name, mode in [("db2", "symmetric"), ("haar", "periodization"), ("bior2.2", "zero")]:
        ca, (ch, cv, cdd) = pywt.dwt2(img, name, mode=mode)
        out.append(
            case(f"dwt2/{name}/{mode}", "dwt2", [img], np.stack([ca, ch, cv, cdd]), **TOL, wavelet=name, mode=mode)
        )
        rec = pywt.idwt2((ca, (ch, cv, cdd)), name, mode=mode)
        out.append(case(f"idwt2/{name}/{mode}", "idwt2", [img], rec, **TOL, wavelet=name, mode=mode))
    c2 = pywt.wavedec2(img, "db1", level=2)
    flat2 = np.concatenate([c2[0].ravel()] + [d.ravel() for lvl in c2[1:] for d in lvl])
    out.append(case("wavedec2", "wavedec2", [img], flat2, **TOL, wavelet="db1", level=2))
    out.append(case("waverec2", "waverec2", [img], pywt.waverec2(c2, "db1"), **TOL, wavelet="db1", level=2))
    for n, name in [(1000, "db4"), (7, "db4"), (64, "haar"), (100, "sym8")]:
        out.append(
            case(
                f"dwtMaxLevel/{n}/{name}",
                "dwtMaxLevel",
                [],
                np.array(float(pywt.dwt_max_level(n, name))),
                n=n,
                wavelet=name,
            )
        )
    return out
