"""Vectorisers against scikit-learn (CountVectorizer with word, char and char_wb analyzers, n-grams, stop words,
document-frequency and size limits; TfidfVectorizer with each normalisation and idf option; HashingVectorizer) and
BM25 / BM25+ against rank_bm25."""

from importlib.metadata import version
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "rank-bm25": version("rank-bm25")}

DOCS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly; the dog sleeps.",
    "A quick brown dog outpaces a quick fox!",
    "Café naïve résumé — the  fox\tand the CAFÉ.",
    "Dogs and foxes: friends or foes?",
]

CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("bigrams", {"ngram_range": (1, 2)}),
    ("only-bigrams", {"ngram_range": (2, 3)}),
    ("stop", {"stop_words": "english", "ngram_range": (1, 2)}),
    ("accents", {"strip_accents": "unicode"}),
    ("ascii", {"strip_accents": "ascii", "lowercase": False}),
    ("df", {"min_df": 2, "max_df": 0.8}),
    ("df-proportion", {"min_df": 0.3}),
    ("max-features", {"max_features": 4}),  # numpy breaks ties arbitrarily: cut where the counts drop
    ("binary", {"binary": True}),
    ("char", {"analyzer": "char", "ngram_range": (2, 3)}),
    ("char-uni", {"analyzer": "char", "ngram_range": (1, 2)}),
    ("char-wb", {"analyzer": "char_wb", "ngram_range": (2, 4)}),
]

CAMEL = {
    "ngram_range": "ngramRange",
    "stop_words": "stopWords",
    "strip_accents": "stripAccents",
    "lowercase": "lowercase",
    "min_df": "minDf",
    "max_df": "maxDf",
    "max_features": "maxFeatures",
    "binary": "binary",
    "analyzer": "analyzer",
    "norm": "norm",
    "use_idf": "useIdf",
    "smooth_idf": "smoothIdf",
    "sublinear_tf": "sublinearTf",
    "n_features": "nFeatures",
    "alternate_sign": "alternateSign",
}


def camel(cfg: dict[str, Any]) -> dict[str, Any]:
    """scikit-learn option names to mdml-js option names (tuples to lists)."""
    return {CAMEL[k]: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


def dense(m: object) -> npt.NDArray[np.float64]:
    """A scikit-learn sparse result as a dense float array."""
    return np.asarray(cast(Any, m).toarray(), dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for name, cfg in CONFIGS:
        cv = CountVectorizer(**cfg)
        counts = dense(cv.fit_transform(DOCS))
        names = [str(f) for f in cv.get_feature_names_out()]
        out.append(case(f"count/{name}", "countVectorize", [DOCS, camel(cfg)], counts, names=names))
        new = dense(cv.transform(["the fox and the dog", "unseen words only"]))
        out.append(case(f"count/transform/{name}", "countTransform", [DOCS, camel(cfg)], new))
    tfidf_configs: list[tuple[str, dict[str, Any]]] = [
        ("default", {}),
        ("l1", {"norm": "l1"}),
        ("none", {"norm": None}),
        ("raw-idf", {"smooth_idf": False}),
        ("no-idf", {"use_idf": False}),
        ("sublinear", {"sublinear_tf": True, "ngram_range": (1, 2)}),
    ]
    for name, cfg in tfidf_configs:
        tv = TfidfVectorizer(**cfg)
        weights = dense(tv.fit_transform(DOCS))
        opts = camel(cfg)
        if "norm" in opts and opts["norm"] is None:
            opts["norm"] = "none"
        out.append(case(f"tfidf/{name}", "tfidfVectorize", [DOCS, opts], weights, rtol=1e-12, atol=1e-14))
        if tv.use_idf:
            out.append(case(f"tfidf/idf/{name}", "idf", [DOCS, opts], np.asarray(tv.idf_), rtol=1e-12, atol=1e-14))
        new = dense(tv.transform(["the quick fox", "unseen"]))
        out.append(case(f"tfidf/transform/{name}", "tfidfTransform", [DOCS, opts], new, rtol=1e-12, atol=1e-14))
    hashing_configs: list[tuple[str, dict[str, Any]]] = [
        ("default", {"n_features": 16}),
        ("no-sign", {"n_features": 32, "alternate_sign": False, "norm": None}),
        ("binary", {"n_features": 8, "binary": True, "norm": "l1"}),
        ("char", {"n_features": 64, "analyzer": "char_wb", "ngram_range": (2, 3)}),
    ]
    for name, cfg in hashing_configs:
        hashed = dense(HashingVectorizer(**cfg).transform(DOCS))
        opts = camel(cfg)
        if "norm" in opts and opts["norm"] is None:
            opts["norm"] = "none"
        out.append(case(f"hashing/{name}", "hashingVectorize", [DOCS, opts], hashed, rtol=1e-12, atol=1e-14))
    corpus = [d.lower().replace(".", "").replace(",", "").split() for d in DOCS]
    queries = [["quick", "fox"], ["lazy", "dog", "dog"], ["the"], ["unseen"]]
    for qi, q in enumerate(queries):
        out.append(
            case(f"bm25/okapi/{qi}", "bm25", [corpus, q], np.asarray(BM25Okapi(corpus).get_scores(q)), variant="okapi")
        )
        out.append(
            case(
                f"bm25/okapi-k1/{qi}",
                "bm25",
                [corpus, q],
                np.asarray(BM25Okapi(corpus, k1=1.2, b=0.5, epsilon=0.1).get_scores(q)),
                variant="okapi",
                k1=1.2,
                b=0.5,
                epsilon=0.1,
            )
        )
        out.append(
            case(f"bm25/plus/{qi}", "bm25", [corpus, q], np.asarray(BM25Plus(corpus).get_scores(q)), variant="plus")
        )
    return out
