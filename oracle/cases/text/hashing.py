"""MurmurHash3 against scikit-learn's murmurhash3_32, and fastText's FNV-1a (sign-extended bytes) against gensim's
ft_hash_bytes; the plain FNV-1a is checked against its definition."""

from importlib.metadata import version

from gensim.models.fasttext import ft_hash_bytes
from sklearn.utils import murmurhash3_32
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "gensim": version("gensim")}

KEYS: list[str] = [
    "",
    "a",
    "ab",
    "abc",
    "abcd",
    "abcde",
    "foo",
    "hello world",
    "naïve",
    "日本語",
    "😀",
    "The quick brown fox",
]


def fnv1a(data: bytes) -> int:
    """The 32-bit FNV-1a hash of bytes."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def cases() -> list[Case]:
    out: list[Case] = []
    for k in KEYS:
        out.append(case(f"murmur/{k}", "murmurhash3", [k], [int(murmurhash3_32(k, seed=s)) for s in [0, 1, 42]]))
        out.append(case(f"murmur/positive/{k}", "murmurhash3Positive", [k], int(murmurhash3_32(k, positive=True))))
        out.append(case(f"fnv/{k}", "fnv1a", [k], fnv1a(k.encode("utf-8"))))
        out.append(case(f"fasttext/{k}", "fnv1aSigned", [k], int(ft_hash_bytes(k.encode("utf-8")))))
    return out
