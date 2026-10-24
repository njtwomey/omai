"""WordNet's morphy and nltk's WordNetLemmatizer against nltk, with a small lexicon and exception lists installed in
a reader instance (so no WordNet download is needed): the rules, the exception path, the filtering and the choice of
the shortest lemma all come from nltk's code."""

from importlib.metadata import version

from nltk.corpus.reader.wordnet import WordNetCorpusReader
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

LEMMAS: dict[str, list[str]] = {
    "n": ["dog", "church", "churche", "bus", "box", "wolf", "woman", "pony", "axis", "axe", "glass", "fly", "news"],
    "v": ["run", "be", "go", "fly", "die", "hope", "hop", "make", "agree", "agre", "stop", "use"],
    "a": ["big", "fast", "late", "good", "nice"],
    "r": ["fast", "well"],
}
EXCEPTIONS: dict[str, dict[str, list[str]]] = {
    "n": {"axes": ["axis", "axe"], "mice": ["mouse"]},
    "v": {"went": ["go"], "was": ["be"], "ran": ["run"], "dying": ["die"]},
    "a": {"better": ["good"]},
    "r": {"better": ["well"]},
}
WORDS: list[tuple[str, str]] = [
    ("dogs", "n"),
    ("churches", "n"),
    ("buses", "n"),
    ("boxes", "n"),
    ("wolves", "n"),
    ("women", "n"),
    ("ponies", "n"),
    ("axes", "n"),
    ("mice", "n"),
    ("glasses", "n"),
    ("flies", "n"),
    ("news", "n"),
    ("dog", "n"),
    ("unknowns", "n"),
    ("running", "v"),
    ("went", "v"),
    ("was", "v"),
    ("flies", "v"),
    ("dying", "v"),
    ("hoping", "v"),
    ("hopped", "v"),
    ("makes", "v"),
    ("agreed", "v"),
    ("stopped", "v"),
    ("used", "v"),
    ("bigger", "a"),
    ("fastest", "a"),
    ("later", "a"),
    ("better", "a"),
    ("nicest", "a"),
    ("better", "r"),
    ("fast", "r"),
]


def reader() -> WordNetCorpusReader:
    """A WordNet reader with an in-memory lexicon instead of the corpus files."""
    r = object.__new__(WordNetCorpusReader)
    lemma_map: dict[str, dict[str, list[int]]] = {}
    for pos, words in LEMMAS.items():
        for w in words:
            lemma_map.setdefault(w, {})[pos] = [0]
    exception_map = {pos: {k: list(v) for k, v in EXCEPTIONS.get(pos, {}).items()} for pos in ["n", "v", "a", "r", "s"]}
    r._lemma_pos_offset_map = lemma_map  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    r._exception_map = exception_map  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    return r


def cases() -> list[Case]:
    wn = reader()
    out: list[Case] = []
    for word, pos in WORDS:
        lemmas: list[str] = wn._morphy(word, pos)  # pyright: ignore[reportPrivateUsage]
        lemma = min(lemmas, key=len) if lemmas else word
        out.append(case(f"morphy/{pos}/{word}", "morphy", [word, pos], lemmas))
        out.append(case(f"lemmatize/{pos}/{word}", "lemmatize", [word, pos], lemma))
    subs = {pos: [list(r) for r in rules] for pos, rules in WordNetCorpusReader.MORPHOLOGICAL_SUBSTITUTIONS.items()}
    out.append(case("substitutions", "substitutions", [], subs))
    return out
