"""BPE training and encoding against HF tokenizers' BpeTrainer and models.BPE: character-level with special tokens,
a frequency floor and an alphabet limit, byte-level (GPT-2's alphabet), and with a continuing-subword prefix or an
end-of-word suffix. Training text is pre-tokenised by HF's Whitespace (or ByteLevel) pre-tokeniser; the fixture keeps
the vocabulary in id order, the merges, and the model's tokenisation (tokens, ids, offsets) of held-out words.

HF numbers prefixed and suffixed symbols in hash-map order, which also breaks ties between pairs of them, so those
cases keep the vocabulary sorted and tokenisations without ids, and stop before the first tie whose resolution could
depend on the hash state. The alphabet limit is likewise set where the character counts strictly drop."""

import json
from importlib.metadata import version
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.text import tie_free_vocab_size

REFERENCE: dict[str, str] = {"tokenizers": version("tokenizers")}

CORPUS: list[str] = [
    "low lower lowest newer wider new news",
    "the quick brown fox jumps over the lazy dog",
    "lowly newest widest renew knew, the newer news!",
    "naïve café owners serve crème brûlée to lowly newcomers",
    "Dogs don't jump; foxes do. The dog's owner knew it.",
] * 2 + ["low low low lower lower newest newest newest newest widest widest"]

HELD_OUT: list[str] = ["lowering", "newsworthy", "brûlée", "Thequick", "z", "jumped", "日本"]

CONFIGS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("char", {"vocab_size": 80, "special_tokens": ["[UNK]", "[PAD]"]}, {}),
    ("min-frequency", {"vocab_size": 200, "min_frequency": 4, "special_tokens": ["[UNK]"]}, {}),
    ("limit-alphabet", {"vocab_size": 70, "limit_alphabet": 0, "special_tokens": ["[UNK]"]}, {}),
    (
        "prefix",
        {"vocab_size": 90, "continuing_subword_prefix": "##", "special_tokens": ["[UNK]"]},
        {"continuing_subword_prefix": "##"},
    ),
    (
        "suffix",
        {"vocab_size": 90, "end_of_word_suffix": "</w>", "special_tokens": ["[UNK]"]},
        {"end_of_word_suffix": "</w>"},
    ),
]


def train(trainer_cfg: dict[str, Any], model_cfg: dict[str, Any], byte_level: bool) -> Tokenizer:
    """Train a BPE tokenizer with HF's Whitespace (or ByteLevel) pre-tokeniser."""
    tok = Tokenizer(models.BPE(unk_token=None if byte_level else "[UNK]", **model_cfg))
    if byte_level:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), **trainer_cfg
        )
    else:
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(show_progress=False, **trainer_cfg)
    tok.train_from_iterator(CORPUS, trainer)
    return tok


def encodings(tok: Tokenizer, byte_level: bool) -> list[list[list[Any]]]:
    """The model's tokenisation of each held-out word: token, id and offsets."""
    out: list[list[list[Any]]] = []
    for w in HELD_OUT:
        word = w
        if byte_level:
            word = "".join(p for p, _ in pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(w))
        out.append([[t.value, t.id, list(t.offsets)] for t in tok.model.tokenize(word)])
    return out


def merges_of(tok: Tokenizer) -> list[list[str]]:
    """The merges of a trained BPE model, as pairs."""
    raw = json.loads(tok.to_str())["model"]["merges"]
    return [list(m) if isinstance(m, list) else str(m).split(" ") for m in raw]


def strict_alphabet_limit(target: int) -> int:
    """The largest limit at most `target` that does not cut between characters of equal count."""
    counts: dict[str, int] = {}
    for line in CORPUS:
        for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(line):
            for ch in w:
                counts[ch] = counts.get(ch, 0) + 1
    ordered = sorted(counts.values(), reverse=True)
    limit = target
    while limit > 1 and ordered[limit - 1] == ordered[limit]:
        limit -= 1
    return limit


def word_counts() -> dict[str, int]:
    """The words of the corpus as HF's Whitespace pre-tokeniser finds them, with counts."""
    counts: dict[str, int] = {}
    for line in CORPUS:
        for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(line):
            counts[w] = counts.get(w, 0) + 1
    return counts


def cases() -> list[Case]:
    out: list[Case] = []
    for name, cfg, model_cfg in [
        *CONFIGS,
        ("byte-level", {"vocab_size": 300, "special_tokens": ["<|endoftext|>"]}, {}),
    ]:
        byte_level = name == "byte-level"
        affixed = bool(model_cfg)
        trainer_cfg: dict[str, Any] = dict(cfg)
        if name == "limit-alphabet":
            trainer_cfg["limit_alphabet"] = strict_alphabet_limit(30)
        if affixed:
            trainer_cfg["vocab_size"] = tie_free_vocab_size(
                word_counts(),
                model_cfg.get("continuing_subword_prefix"),
                model_cfg.get("end_of_word_suffix"),
                len(trainer_cfg["special_tokens"]),
                int(trainer_cfg["vocab_size"]),
            )
        tok = train(trainer_cfg, model_cfg, byte_level)
        vocab = tok.get_vocab()
        tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        merges = merges_of(tok)
        held = encodings(tok, byte_level)
        if affixed:
            tokens = sorted(tokens)
            held = [[[t, -1, offsets] for t, _, offsets in e] for e in held]
        options = {
            "vocabSize": trainer_cfg["vocab_size"],
            "specialTokens": trainer_cfg.get("special_tokens", []),
            "minFrequency": trainer_cfg.get("min_frequency", 0),
            **({"limitAlphabet": trainer_cfg["limit_alphabet"]} if "limit_alphabet" in trainer_cfg else {}),
            **(
                {"continuingSubwordPrefix": model_cfg["continuing_subword_prefix"]}
                if "continuing_subword_prefix" in model_cfg
                else {}
            ),
            **({"endOfWordSuffix": model_cfg["end_of_word_suffix"]} if "end_of_word_suffix" in model_cfg else {}),
            **({"byteLevel": True} if byte_level else {}),
        }
        out.append(
            case(
                f"bpe/{name}",
                "trainBpe",
                [CORPUS, options],
                tokens,
                merges=merges,
                heldOut=HELD_OUT,
                encodings=held,
                affixed=affixed,
            )
        )
    return out
