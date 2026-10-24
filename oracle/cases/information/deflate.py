"""DEFLATE, zlib and gzip byte for byte against Python's zlib and gzip modules.

Inputs are rebuilt on the TypeScript side from a small recipe (an integer generator that both languages compute
exactly), so the fixture stores only the compressed output: in full for small inputs, and as its CRC-32 and length for
the large ones that slide the window and span several blocks.
"""

import gzip
import io
import zlib

import numpy as np
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"zlib": zlib.ZLIB_RUNTIME_VERSION}

VOCAB = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "for",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "I",
    "this",
    "are",
    "or",
    "his",
    "from",
    "at",
    "which",
    "but",
    "have",
    "an",
    "had",
    "they",
    "you",
    "were",
    "their",
    "one",
    "all",
    "we",
    "can",
    "her",
    "has",
    "there",
    "been",
    "if",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "information",
    "entropy",
    "channel",
    "code",
    "source",
    "noise",
    "message",
    "bits",
]

STRATEGIES: dict[str, int] = {
    "filtered": zlib.Z_FILTERED,
    "huffmanOnly": zlib.Z_HUFFMAN_ONLY,
    "rle": zlib.Z_RLE,
    "fixed": zlib.Z_FIXED,
}


def lcg(x: int) -> int:
    """The generator shared with the tests: x -> (75 x + 74) mod 65537."""
    return (75 * x + 74) % 65537


def recipe(kind: str, n: int, seed: int) -> bytes:
    """The bytes of a recipe: pseudo-English text, pseudo-random bytes or runs of repeated letters."""
    out = bytearray()
    x = seed
    while len(out) < n:
        x = lcg(x)
        if kind == "text":
            word = VOCAB[x % len(VOCAB)]
            x = lcg(x)
            sep = x % 16
            out += (word + (" " if sep < 12 else ", " if sep < 14 else ".\n" if sep == 14 else "\n")).encode()
        elif kind == "random":
            out.append((x >> 3) & 0xFF)
        else:
            byte = 97 + x % 7
            x = lcg(x)
            out += bytes([byte]) * (1 + x % 40)
    return bytes(out[:n])


def as_array(b: bytes) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.frombuffer(b, dtype=np.uint8).astype(np.int64)


def digest(b: bytes) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """CRC-32 and length: an exactness check for outputs too large to store (as floats: CRCs overflow int32)."""
    return np.array([zlib.crc32(b), len(b)], dtype=np.float64)


def zlib_with(data: bytes, level: int, strategy: int, wbits: int) -> bytes:
    c = zlib.compressobj(level, zlib.DEFLATED, wbits, 8, strategy)
    return c.compress(data) + c.flush()


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [0, 1, 2, 3, 10, 300, 3000]:
        data = recipe("text", n, 7)
        for level in range(10):
            out.append(
                case(
                    f"zlib/text/n={n}/level={level}",
                    "zlib",
                    [],
                    as_array(zlib.compress(data, level)),
                    kind="text",
                    n=n,
                    seed=7,
                    level=level,
                )
            )
    big: list[tuple[str, int, int, list[int]]] = [
        ("text", 80000, 11, [1, 3, 4, 6, 9]),
        ("random", 20000, 5, [1, 6]),
        ("runs", 50000, 3, [1, 6, 9]),
    ]
    for kind, n, seed, levels in big:
        data = recipe(kind, n, seed)
        for level in levels:
            out.append(
                case(
                    f"zlib/{kind}/n={n}/level={level}/digest",
                    "zlibDigest",
                    [],
                    digest(zlib.compress(data, level)),
                    kind=kind,
                    n=n,
                    seed=seed,
                    level=level,
                )
            )
    for name, strategy in STRATEGIES.items():
        for kind, n in [("text", 3000), ("runs", 3000)]:
            data = recipe(kind, n, 9)
            out.append(
                case(
                    f"zlib/{kind}/strategy={name}",
                    "zlib",
                    [],
                    as_array(zlib_with(data, 6, strategy, 15)),
                    kind=kind,
                    n=n,
                    seed=9,
                    level=6,
                    strategy=name,
                )
            )
        data = recipe("text", 60000, 21)
        out.append(
            case(
                f"zlib/text/n=60000/strategy={name}/digest",
                "zlibDigest",
                [],
                digest(zlib_with(data, 6, strategy, 15)),
                kind="text",
                n=60000,
                seed=21,
                level=6,
                strategy=name,
            )
        )
    text = recipe("text", 3000, 4)
    for level in [1, 6, 9]:
        out.append(
            case(
                f"deflate/raw/level={level}",
                "deflate",
                [],
                as_array(zlib_with(text, level, zlib.Z_DEFAULT_STRATEGY, -15)),
                kind="text",
                n=3000,
                seed=4,
                level=level,
            )
        )
        out.append(
            case(
                f"gzip/level={level}",
                "gzip",
                [],
                as_array(gzip.compress(text, compresslevel=level, mtime=1700000000)),
                kind="text",
                n=3000,
                seed=4,
                level=level,
                mtime=1700000000,
            )
        )
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="notes.txt", mode="wb", compresslevel=6, fileobj=buffer, mtime=1) as f:
        f.write(text)
    out.append(
        case(
            "gzip/name",
            "gzip",
            [],
            as_array(buffer.getvalue()),
            kind="text",
            n=3000,
            seed=4,
            level=6,
            mtime=1,
            filename="notes.txt",
        )
    )
    two = gzip.compress(b"first member, ", mtime=1) + gzip.compress(b"second member", mtime=1)
    out.append(case("gzip/members", "gunzip", [as_array(two)], as_array(b"first member, second member")))
    for kind, n, seed in [("text", 5000, 2), ("random", 1000, 3)]:
        data = recipe(kind, n, seed)
        out.append(
            case(
                f"checksums/{kind}",
                "checksums",
                [],
                np.array([zlib.crc32(data), zlib.adler32(data)], dtype=np.float64),
                kind=kind,
                n=n,
                seed=seed,
            )
        )
    return out
