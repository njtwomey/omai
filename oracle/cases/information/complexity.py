"""Compression distance, Lempel-Ziv complexity, entropy rates and approximate and sample entropy against zlib,
antropy, scipy and numpy."""

import importlib.metadata
import zlib
from collections import Counter

import antropy  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import scipy
import scipy.linalg as sla
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "zlib": zlib.ZLIB_RUNTIME_VERSION,
    "antropy": importlib.metadata.version("antropy"),
    "scipy": scipy.__version__,
}


def ncd(x: bytes, y: bytes) -> float:
    cx, cy, cxy = (len(zlib.compress(b, 9)) for b in (x, y, x + y))
    return (cxy - min(cx, cy)) / max(cx, cy)


def block_entropy(seq: list[int], length: int) -> float:
    blocks = Counter(tuple(seq[i : i + length]) for i in range(len(seq) - length + 1))
    return float(st.entropy(list(blocks.values())))


def markov_rate(p: FloatArray) -> float:
    pi = sla.null_space(p.T - np.eye(p.shape[0]))[:, 0]
    pi = pi / pi.sum()
    return float(np.sum(pi * st.entropy(p, axis=1)))


def as_bytes(s: str) -> FloatArray:
    return np.frombuffer(s.encode(), dtype=np.uint8).astype(np.float64)


def cases() -> list[Case]:
    g = rng(460)
    out: list[Case] = []
    texts = [
        "the quick brown fox jumps over the lazy dog " * 6,
        "the quick brown cat sleeps under the warm sun " * 6,
        "".join(chr(97 + int(v)) for v in g.integers(0, 26, size=300)),
    ]
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i <= j:
                out.append(
                    case(
                        f"ncd/{i}-{j}",
                        "normalisedCompressionDistance",
                        [as_bytes(a), as_bytes(b)],
                        np.array(ncd(a.encode(), b.encode())),
                    )
                )
    binary = [int(v) for v in g.integers(0, 2, size=200)]
    words = "HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD!"
    for name, seq in [("binary", binary), ("periodic", [0, 1, 1] * 30), ("letters", [ord(c) for c in words])]:
        arr = np.array(seq, dtype=np.float64)
        out.append(
            case(f"lz/{name}", "lempelZivComplexity", [arr], np.array(float(antropy.lziv_complexity(np.array(seq)))))
        )
        norm = float(antropy.lziv_complexity(np.array(seq), normalize=True))
        out.append(case(f"lz/{name}/normalised", "lempelZivComplexity", [arr], np.array(norm), normalise=True))
    markov = [0]
    p = np.array([[0.9, 0.1], [0.4, 0.6]])
    for _ in range(400):
        markov.append(int(g.random() < p[markov[-1], 1]))
    for length in [1, 2, 3]:
        out.append(
            case(
                f"blockEntropy/{length}",
                "blockEntropy",
                [np.array(markov, dtype=np.float64)],
                np.array(block_entropy(markov, length)),
                length=length,
            )
        )
    p3 = g.uniform(0.1, 1, size=(3, 3))
    p3 = p3 / p3.sum(axis=1, keepdims=True)
    out.append(case("markovEntropyRate/2", "markovEntropyRate", [p], np.array(markov_rate(p)), rtol=1e-10))
    out.append(case("markovEntropyRate/3", "markovEntropyRate", [p3], np.array(markov_rate(p3)), rtol=1e-10))
    t = np.arange(300)
    series = {
        "sine": np.sin(2 * np.pi * t / 25.0),
        "noise": g.normal(size=300),
        "ar1": np.zeros(300),
    }
    for i in range(1, 300):
        series["ar1"][i] = 0.8 * series["ar1"][i - 1] + g.normal()
    for name, x in series.items():
        for order in [2, 3]:
            out.append(
                case(
                    f"approximateEntropy/{name}/m={order}",
                    "approximateEntropy",
                    [x],
                    np.array(antropy.app_entropy(x, order=order)),
                    order=order,
                )
            )
            out.append(
                case(
                    f"sampleEntropy/{name}/m={order}",
                    "sampleEntropy",
                    [x],
                    np.array(antropy.sample_entropy(x, order=order)),
                    order=order,
                )
            )
        out.append(
            case(
                f"sampleEntropy/{name}/r=0.5",
                "sampleEntropy",
                [x],
                np.array(antropy.sample_entropy(x, order=2, tolerance=0.5)),
                order=2,
                tolerance=0.5,
            )
        )
    return out
