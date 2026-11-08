"""MinHash signatures, Jaccard estimates and LSH against datasketch (its default affine32 scheme, with scikit-learn's
MurmurHash3 as the item hash so the signatures are reproducible): the permutations come from datasketch, the
optimal banding from MinHashLSH's parameter search, and the candidates from MinHashLSH queries."""

from importlib.metadata import version

import numpy as np
from datasketch import MinHash, MinHashLSH
from omai_oracle.fixture import Case, case
from sklearn.utils import murmurhash3_32

REFERENCE: dict[str, str] = {"datasketch": version("datasketch"), "scikit-learn": version("scikit-learn")}

DOCS: list[str] = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumped over the lazy dog",
    "the quick brown fox jumps over the lazy cat",
    "an entirely different sentence about cats and dogs",
    "an entirely different sentence about cats and frogs",
    "",
]


def shingle_set(doc: str, k: int = 4) -> list[str]:
    """Character k-shingles (the whole text when shorter)."""
    chars = list(doc)
    if not chars:
        return []
    if len(chars) < k:
        return [doc]
    return sorted({"".join(chars[i : i + k]) for i in range(len(chars) - k + 1)})


def murmur(b: bytes) -> int:
    return int(murmurhash3_32(b, positive=True))


def cases() -> list[Case]:
    out: list[Case] = []
    num_perm = 64
    sigs: list[MinHash] = []
    for i, d in enumerate(DOCS):
        m = MinHash(num_perm=num_perm, seed=1, hashfunc=murmur)
        for s in shingle_set(d):
            m.update(s.encode("utf-8"))
        sigs.append(m)
        a, b = np.asarray(m.permutations)
        values = np.asarray(m.hashvalues)
        out.append(
            case(
                f"minhash/{i}",
                "minhash",
                [shingle_set(d), [int(x) for x in a], [int(x) for x in b]],
                [int(x) for x in values],
            )
        )
    est = np.array([[s.jaccard(t) for t in sigs[:-1]] for s in sigs[:-1]])
    out.append(case("jaccard", "jaccard", [], est))
    for threshold, perms in [(0.5, 64), (0.8, 64), (0.3, 128), (0.7, 32)]:
        lsh = MinHashLSH(threshold=threshold, num_perm=perms)
        out.append(case(f"lsh/{threshold}/{perms}", "lshParameters", [threshold, perms], [lsh.b, lsh.r]))
    for threshold in [0.5, 0.8]:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for i, m in enumerate(sigs):
            lsh.insert(str(i), m)
        cands = [sorted(int(str(x)) for x in lsh.query(m)) for m in sigs]
        out.append(case(f"query/{threshold}", "lshQuery", [threshold], cands, bands=lsh.b, rows=lsh.r))
    return out
