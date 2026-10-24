"""n-grams, everygrams and skip-grams against nltk; fastText character n-grams and their hash buckets against
gensim's ports of fastText (compute_ngrams_bytes, which follows fastText's order and drops the lone boundary markers,
and ft_ngram_hashes)."""

from importlib.metadata import version

from gensim.models.fasttext import ft_ngram_hashes
from gensim.models.fasttext_inner import compute_ngrams_bytes  # pyright: ignore[reportMissingModuleSource]
from nltk.util import everygrams, ngrams, skipgrams
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"nltk": version("nltk"), "gensim": version("gensim")}

SENTENCES: list[list[str]] = [
    [str(w) for w in "insurgents killed in ongoing fighting".split()],  # noqa: SIM905
    [str(w) for w in "a rose is a rose is a rose".split()],  # noqa: SIM905
    ["x"],
    [],
]
WORDS: list[str] = ["where", "a", "naïve", "日本語", "😀ok", "fastText", "ab"]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, s in enumerate(SENTENCES):
        for n in [1, 2, 3]:
            out.append(case(f"ngrams/{i}/{n}", "ngrams", [s, n], [list(g) for g in ngrams(s, n)]))
            padded = [
                list(g)
                for g in ngrams(s, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
            ]
            out.append(case(f"ngrams/padded/{i}/{n}", "ngramsPadded", [s, n], padded))
        out.append(case(f"everygrams/{i}", "everygrams", [s, 1, 3], [list(g) for g in everygrams(s, 1, 3)]))
        for n, k in [(2, 0), (2, 2), (3, 2)]:
            out.append(case(f"skipgrams/{i}/{n}/{k}", "skipgrams", [s, n, k], [list(g) for g in skipgrams(s, n, k)]))
    for w in WORDS:
        for minn, maxn in [(3, 6), (1, 3), (2, 2)]:
            out.append(
                case(
                    f"fasttext/ngrams/{w}/{minn}-{maxn}",
                    "fastTextNgrams",
                    [w, minn, maxn],
                    [g.decode("utf-8") for g in compute_ngrams_bytes(w, minn, maxn)],
                )
            )
            out.append(
                case(
                    f"fasttext/buckets/{w}/{minn}-{maxn}",
                    "fastTextBuckets",
                    [w, minn, maxn, 2000000],
                    [int(h) for h in ft_ngram_hashes(w, minn, maxn, 2000000)],
                )
            )
    return out
