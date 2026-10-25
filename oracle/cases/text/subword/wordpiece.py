"""WordPiece training and greedy longest-match encoding against HF tokenizers' WordPieceTrainer and models.WordPiece.

The trainer is BPE with the `##` prefix, whose ids for prefixed symbols follow hash-map order, so the fixture keeps
the vocabulary sorted and stops before the first tie whose resolution could depend on the hash state; encodings
use fixed vocabularies (BERT-style) whose ids are known."""

from importlib.metadata import version
from typing import Any

from mdml_oracle.fixture import Case, case
from mdml_oracle.text import tie_free_vocab_size
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

REFERENCE: dict[str, str] = {"tokenizers": version("tokenizers")}

TEXT: str = (
    "the of and to in is was for that with as his on be at by it from had an are not this but which have or were"
    " been one their they has her all would there more if when out who also so them some what into other its about"
    " than then only time may these over first new two most used after made could many like years can very such any"
    " where three must do between through state how even should before being use those while each life well under"
    " own back found world because both great right still same united last"
)
WORDS: list[str] = [str(w) for w in TEXT.split()]
# Zipfian counts (word i about 400 / (i + 3) times), so ties come late.
ZIPF: list[str] = [w for i, w in enumerate(WORDS) for _ in range(max(1, round(400 / (i + 3))))]
CORPUS: list[str] = [" ".join(ZIPF[i : i + 12]) for i in range(0, len(ZIPF), 12)]

VOCAB: list[str] = [
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "un",
    "##aff",
    "##able",
    "a",
    "##ff",
    "##a",
    "able",
    "##b",
    "##le",
    "run",
    "##ning",
    "##n",
    "##er",
    "日",
    "##本",
    "😀",
]


def cases() -> list[Case]:
    out: list[Case] = []
    counts: dict[str, int] = {}
    for line in CORPUS:
        for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(line):
            counts[w] = counts.get(w, 0) + 1
    size = tie_free_vocab_size(counts, "##", None, 2, 200)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=size, special_tokens=["[UNK]", "[CLS]"], show_progress=False)
    tok.train_from_iterator(CORPUS, trainer)
    options: dict[str, Any] = {"vocabSize": size, "specialTokens": ["[UNK]", "[CLS]"]}
    out.append(case("train", "trainWordPiece", [CORPUS, options], sorted(tok.get_vocab())))
    model = models.WordPiece({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]", max_input_chars_per_word=12)
    fixed = Tokenizer(model)
    for w in [
        "unaffable",
        "able",
        "unable",
        "running",
        "runner",
        "aaff",
        "xyz",
        "日本",
        "日本本",
        "😀",
        "abcdefghijklm",
    ]:
        out.append(
            case(
                f"tokenize/{w}",
                "wordPieceTokenize",
                [VOCAB, w],
                [[t.value, t.id, list(t.offsets)] for t in fixed.model.tokenize(w)],
            )
        )
    return out
