"""Collocations against nltk: bigram and trigram finders (with wider windows, frequency and word filters) scored by
every association measure of BigramAssocMeasures and TrigramAssocMeasures (Fisher's test through scipy)."""

from collections.abc import Callable
from importlib.metadata import version

from mdml_oracle.fixture import Case, case
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

REFERENCE: dict[str, str] = {"nltk": version("nltk"), "scipy": version("scipy")}

TEXT: str = (
    "new york is a big city and new york is old . the big apple is new york . "
    "san francisco is far from new york but san francisco is also big . "
    "people in new york like the big city and people in san francisco like the bay . "
    "the bay area is near san francisco . a big city has big problems ."
)
WORDS: list[str] = [str(w) for w in TEXT.split()]

BIGRAM: list[tuple[str, Callable[..., float]]] = [
    ("rawFreq", BigramAssocMeasures.raw_freq),
    ("studentT", BigramAssocMeasures.student_t),
    ("chiSq", BigramAssocMeasures.chi_sq),
    ("pmi", BigramAssocMeasures.pmi),
    ("likelihoodRatio", BigramAssocMeasures.likelihood_ratio),
    ("poissonStirling", BigramAssocMeasures.poisson_stirling),
    ("jaccard", BigramAssocMeasures.jaccard),
    ("miLike", BigramAssocMeasures.mi_like),
    ("dice", BigramAssocMeasures.dice),
    ("phiSq", BigramAssocMeasures.phi_sq),
    ("fisher", BigramAssocMeasures.fisher),
]
TRIGRAM: list[tuple[str, Callable[..., float]]] = [
    ("rawFreq", TrigramAssocMeasures.raw_freq),
    ("studentT", TrigramAssocMeasures.student_t),
    ("chiSq", TrigramAssocMeasures.chi_sq),
    ("pmi", TrigramAssocMeasures.pmi),
    ("likelihoodRatio", TrigramAssocMeasures.likelihood_ratio),
    ("poissonStirling", TrigramAssocMeasures.poisson_stirling),
    ("jaccard", TrigramAssocMeasures.jaccard),
    ("miLike", TrigramAssocMeasures.mi_like),
]


def is_punct(w: str) -> bool:
    return w == "."


def cases() -> list[Case]:
    out: list[Case] = []
    for window in [2, 4]:
        for name, fn in BIGRAM:
            finder = BigramCollocationFinder.from_words(WORDS, window_size=window)
            if window == 4:
                finder.apply_freq_filter(2)
                finder.apply_word_filter(is_punct)
            scored = [[list(ng), float(s)] for ng, s in finder.score_ngrams(fn)]
            out.append(
                case(
                    f"bigram/{window}/{name}",
                    "bigram",
                    [WORDS],
                    scored,
                    rtol=1e-10,
                    atol=1e-12,
                    measure=name,
                    window=window,
                )
            )
    for window in [3, 4]:
        for name, fn in TRIGRAM:
            finder = TrigramCollocationFinder.from_words(WORDS, window_size=window)
            finder.apply_freq_filter(2)
            scored = [[list(ng), float(s)] for ng, s in finder.score_ngrams(fn)]
            out.append(
                case(
                    f"trigram/{window}/{name}",
                    "trigram",
                    [WORDS],
                    scored,
                    rtol=1e-10,
                    atol=1e-12,
                    measure=name,
                    window=window,
                )
            )
    return out
