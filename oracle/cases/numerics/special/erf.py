"""Error functions against mpmath."""

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def erfcx(x: Mpf) -> Mpf:
    return mp.exp(x * x) * mp.erfc(x)


def erfinv(y: Mpf) -> Mpf:
    return mp.erfinv(y)


def erfcinv(y: Mpf) -> Mpf:
    return mp.erfinv(1 - y)


def d_erfinv(y: Mpf) -> Mpf:
    return mp.sqrt(mp.pi) / 2 * mp.exp(erfinv(y) ** 2)


def d_erfcinv(y: Mpf) -> Mpf:
    return -mp.sqrt(mp.pi) / 2 * mp.exp(erfcinv(y) ** 2)


def dawsn(x: Mpf) -> Mpf:
    return mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)


def cases() -> list[Case]:
    g = rng(1)
    bulk = np.concatenate([g.uniform(-3, 3, 30), [0.0, 0.5, 1.5, -1.5]])
    wide = np.concatenate([g.uniform(-27, 27, 20), np.exp(g.uniform(np.log(1e-12), 0, 10))])
    out: list[Case] = []
    for name, f in [("erf", mp.erf), ("erfc", mp.erfc), ("erfcx", erfcx)]:
        out += value_and_grads(f"{name}/bulk", name, f, [bulk], wrt=[0])
        out += value_and_grads(f"{name}/wide", name, f, [wide], wrt=[])
    out += value_and_grads("erfcx/large", "erfcx", erfcx, [np.exp(g.uniform(1, 30, 20))], wrt=[0])
    y = np.concatenate([g.uniform(-0.999, 0.999, 30), 1 - np.exp(g.uniform(np.log(1e-15), np.log(1e-3), 10))])
    out += value_and_grads(
        "erfinv/bulk", "erfinv", erfinv, [y], wrt=[0], rtol=1e-13, atol=1e-16, grad_rtol=1e-12, exact={0: d_erfinv}
    )
    yc = np.concatenate([g.uniform(1e-3, 2 - 1e-3, 30), np.exp(g.uniform(np.log(1e-300), np.log(1e-3), 10))])
    out += value_and_grads(
        "erfcinv/bulk",
        "erfcinv",
        erfcinv,
        [yc],
        wrt=[0],
        rtol=1e-13,
        atol=1e-16,
        grad_rtol=1e-12,
        exact={0: d_erfcinv},
        dps=400,
    )
    out += value_and_grads("dawsn/bulk", "dawsn", dawsn, [np.concatenate([bulk, g.uniform(-20, 20, 20)])], wrt=[0])
    return out
