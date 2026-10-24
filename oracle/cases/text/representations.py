"""Distributional representations: co-occurrence counts (window, harmonic weighting) and PMI / PPMI / shifted PPMI with
context smoothing from their numpy definitions, LSA and SVD word vectors against numpy.linalg.svd (signs fixed so each
left singular vector's largest entry is positive), and nearest words and analogies (3CosAdd, 3CosMul) against gensim's
KeyedVectors."""

from importlib.metadata import version
from typing import Any, cast

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "numpy": version("numpy"),
    "gensim": version("gensim"),
    "scikit-learn": version("scikit-learn"),
}

SENTENCES: list[list[str]] = [
    [str(w) for w in s.split()]
    for s in [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog are pets",
        "the king and the queen rule",
        "a man and a woman walk",
        "the cat chased the dog",
    ]
]


def cooc(window: int, harmonic: bool) -> tuple[list[str], FloatArray]:
    words = sorted({w for s in SENTENCES for w in s})
    idx = {w: i for i, w in enumerate(words)}
    m = np.zeros((len(words), len(words)))
    for s in SENTENCES:
        for i, w in enumerate(s):
            for d in range(1, window + 1):
                for j in (i - d, i + d):
                    if 0 <= j < len(s):
                        m[idx[w], idx[s[j]]] += 1.0 / d if harmonic else 1.0
    return words, m


def pmi(m: FloatArray, alpha: float, positive: bool, shift: float) -> FloatArray:
    total = m.sum()
    pw = m.sum(axis=1, keepdims=True) / total
    col = m.sum(axis=0, keepdims=True) ** alpha
    pc = col / col.sum()
    with np.errstate(divide="ignore"):
        v = np.log((m / total) / (pw * pc)) - np.log(shift)
    return np.maximum(v, 0) if positive else v


def fixed(u: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Flip each singular pair so the left vector's largest-magnitude entry is positive."""
    idx = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[idx, np.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs, v * signs[:, None]


def cases() -> list[Case]:
    out: list[Case] = []
    for window, harmonic in [(1, False), (2, False), (2, True)]:
        words, m = cooc(window, harmonic)
        out.append(
            case(
                f"cooccurrence/{window}/{harmonic}",
                "cooccurrence",
                [SENTENCES],
                m,
                words=words,
                window=window,
                harmonic=harmonic,
            )
        )
        for alpha, positive, shift in [(1.0, False, 1.0), (1.0, True, 1.0), (0.75, True, 1.0), (0.75, True, 2.0)]:
            out.append(
                case(
                    f"pmi/{window}/{harmonic}/{alpha}/{positive}/{shift}",
                    "pmi",
                    [m],
                    pmi(m, alpha, positive, shift),
                    rtol=1e-12,
                    atol=1e-12,
                    alpha=alpha,
                    positive=positive,
                    shift=shift,
                )
            )
    docs = [" ".join(s) for s in SENTENCES]
    x = np.asarray(cast(Any, TfidfVectorizer().fit_transform(docs)).toarray(), dtype=np.float64)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    u, vt = fixed(u, vt)
    k = 3
    out.append(
        case(
            "lsa",
            "lsa",
            [x],
            np.concatenate([s[:k], (u[:, :k] * s[:k]).ravel(), (vt[:k].T * s[:k]).ravel()]),
            rtol=1e-10,
            atol=1e-10,
            k=k,
        )
    )
    _, m = cooc(2, False)
    p = pmi(m, 0.75, True, 1.0)
    u, s, vt = np.linalg.svd(p, full_matrices=False)
    u, vt = fixed(u, vt)
    out.append(case("svd-vectors", "svdWordVectors", [p], u[:, :4] * s[:4] ** 0.5, rtol=1e-9, atol=1e-10, k=4))
    g = rng(77)
    vocab = ["man", "woman", "king", "queen", "apple", "pear", "paris", "france", "rome", "italy"]
    vecs = g.normal(size=(len(vocab), 6))
    vecs[3] = vecs[2] - vecs[0] + vecs[1] + 0.1 * g.normal(size=6)
    vecs[9] = vecs[8] - vecs[6] + vecs[7] + 0.1 * g.normal(size=6)
    kv = KeyedVectors(vector_size=6)
    kv.add_vectors(vocab, vecs)
    near = [[w, float(sc)] for w, sc in kv.most_similar("king", topn=4)]
    add = [[w, float(sc)] for w, sc in kv.most_similar(positive=["woman", "king"], negative=["man"], topn=3)]
    mul = [[w, float(sc)] for w, sc in kv.most_similar_cosmul(positive=["woman", "king"], negative=["man"], topn=3)]
    add2 = [[w, float(sc)] for w, sc in kv.most_similar(positive=["france", "rome"], negative=["paris"], topn=3)]
    out.append(
        case(
            "vectors",
            "vectors",
            [vocab, vecs],
            {"near": near, "add": add, "mul": mul, "add2": add2},
            rtol=1e-6,
            atol=1e-6,
        )
    )
    return out
