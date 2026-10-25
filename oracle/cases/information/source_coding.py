"""Huffman codes against bitarray: the expected codeword length, which every optimal code shares even when ties
make the codeword lengths differ."""

import importlib.metadata

import numpy as np
from bitarray.util import huffman_code
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"bitarray": importlib.metadata.version("bitarray")}


def cases() -> list[Case]:
    g = rng(430)
    out: list[Case] = []
    for i, m in enumerate([2, 3, 5, 8, 13, 26, 40]):
        freq = g.integers(1, 100, size=m)
        if i % 2 == 1:
            freq[: m // 3] = g.integers(1, 3, size=m // 3)
        code = huffman_code({s: int(f) for s, f in enumerate(freq)})
        mean = sum(int(f) * len(code[s]) for s, f in enumerate(freq)) / int(freq.sum())
        out.append(case(f"huffman/expectedLength/m={m}", "huffmanExpectedLength", [freq], np.array(mean)))
    # Dyadic probabilities: the Huffman code meets the entropy exactly.
    dyadic = np.array([32, 16, 8, 4, 2, 1, 1])
    code = huffman_code({s: int(f) for s, f in enumerate(dyadic)})
    mean = sum(int(f) * len(code[s]) for s, f in enumerate(dyadic)) / int(dyadic.sum())
    out.append(case("huffman/expectedLength/dyadic", "huffmanExpectedLength", [dyadic], np.array(mean)))
    return out
