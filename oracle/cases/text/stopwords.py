"""English stop-word lists against scikit-learn's ENGLISH_STOP_WORDS and gensim's STOPWORDS."""

from importlib.metadata import version

from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "gensim": version("gensim")}


def cases() -> list[Case]:
    return [
        case("sklearn", "stopWords", [], sorted(ENGLISH_STOP_WORDS), list="sklearn"),
        case("gensim", "stopWords", [], sorted(STOPWORDS), list="gensim"),
    ]
