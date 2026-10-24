"""Reed-Solomon encoding and decoding (errors and erasures) against reedsolo."""

import importlib.metadata
from collections.abc import Iterable

import numpy as np
import reedsolo
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"reedsolo": importlib.metadata.version("reedsolo")}

# (nsym, message length, bits per symbol, primitive polynomial, first consecutive root)
CODES: list[tuple[int, int, int, int, int]] = [
    (4, 12, 8, 0x11D, 0),
    (10, 40, 8, 0x11D, 0),
    (32, 223, 8, 0x11D, 0),
    (8, 30, 8, 0x11D, 1),
    (4, 9, 4, 0x13, 0),
    (6, 20, 5, 0x25, 0),
]


def as_array(b: Iterable[int]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.array(list(b), dtype=np.int64)


def cases() -> list[Case]:
    g = rng(440)
    out: list[Case] = []
    for nsym, k, m, prim, fcr in CODES:
        size = 1 << m
        codec = reedsolo.RSCodec(nsym, nsize=size - 1, fcr=fcr, prim=prim, c_exp=m)
        message = [int(v) for v in g.integers(0, size, size=k)]
        word = list(codec.encode(bytearray(message)))
        params: dict[str, int] = {"nsym": nsym, "m": m, "primitive": prim, "fcr": fcr}
        tag = f"nsym={nsym}/k={k}/m={m}/fcr={fcr}"
        out.append(case(f"encode/{tag}", "rsEncode", [as_array(message)], as_array(word), **params))
        n = len(word)
        for errors, erasures in [(nsym // 2, 0), (nsym // 4, nsym - 2 * (nsym // 4)), (0, nsym), (1, 0)]:
            positions = [int(p) for p in g.choice(n, size=errors + erasures, replace=False)]
            corrupt = list(word)
            for p in positions:
                corrupt[p] ^= int(g.integers(1, size))
            erase = positions[errors:]
            decoded = codec.decode(bytearray(corrupt), erase_pos=erase if erase else None)
            out.append(
                case(
                    f"decode/{tag}/errors={errors}/erasures={erasures}",
                    "rsDecode",
                    [as_array(corrupt), as_array(erase)],
                    as_array(decoded[1]),
                    **params,
                )
            )
    return out
