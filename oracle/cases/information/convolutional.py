"""Convolutional encoding against scikit-commpy (its Viterbi decoder is not used: it fails under numpy 2, so the
decoder is checked against brute-force maximum likelihood in the TypeScript tests).

mdml-js writes generators with the most significant bit on the current input (as MATLAB's poly2trellis and most
textbooks); commpy puts the current input on the least significant bit, so its generators are the bit reversals."""

import importlib.metadata
import warnings

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import commpy.channelcoding.convcode as cc  # pyright: ignore[reportMissingTypeStubs]

REFERENCE: dict[str, str] = {"scikit-commpy": importlib.metadata.version("scikit-commpy")}

CODES: list[list[int]] = [[0o7, 0o5], [0o171, 0o133], [0o5, 0o7, 0o7], [0o13, 0o15, 0o17]]


def cases() -> list[Case]:
    g = rng(450)
    out: list[Case] = []
    for gens in CODES:
        k = max(int(x).bit_length() for x in gens)
        reversed_gens = [int(f"{x:0{k}b}"[::-1], 2) for x in gens]
        trellis = cc.Trellis(np.array([k - 1]), np.array([reversed_gens]))
        for length in [1, 7, 40]:
            message = g.integers(0, 2, size=length)
            coded = np.asarray(cc.conv_encode(message, trellis), dtype=np.int64)
            name = "/".join(oct(x) for x in gens)
            out.append(
                case(
                    f"encode/{name}/length={length}",
                    "convolutionalEncode",
                    [message.astype(np.int64)],
                    coded,
                    generators=[int(x) for x in gens],
                )
            )
    return out
