"""QR factorisation and least squares against numpy.linalg."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(24)
    out: list[Case] = []
    for m, n in [(5, 3), (4, 4), (3, 5)]:
        a = g.normal(size=(m, n))
        q, r = np.linalg.qr(a)
        out.append(case(f"qr/{m}x{n}/q", "qr", [a], q, rtol=1e-10, atol=1e-13, part="q"))
        out.append(case(f"qr/{m}x{n}/r", "qr", [a], r, rtol=1e-10, atol=1e-13, part="r"))
    batch = g.normal(size=(2, 4, 3))
    out.append(case("qr/batch/r", "qr", [batch], np.linalg.qr(batch)[1], rtol=1e-10, atol=1e-13, part="r"))
    tall, b = g.normal(size=(6, 3)), g.normal(size=6)
    wide, bw = g.normal(size=(3, 5)), g.normal(size=3)
    out.append(case("lstsq/tall", "lstsq", [tall, b], np.linalg.lstsq(tall, b)[0], rtol=1e-10))
    out.append(case("lstsq/wide", "lstsq", [wide, bw], np.linalg.lstsq(wide, bw)[0], rtol=1e-10))
    return out
