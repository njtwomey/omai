"""Unigram language-model training and Viterbi encoding against HF tokenizers' UnigramTrainer and models.Unigram, on
words pre-tokenised by HF's Metaspace pre-tokeniser.

HF sums EM expectations in hash-map order and orders required characters of equal score by hash, so the fixture keeps
the pieces sorted by score and then by piece, rounded to 1e-7 (HF varies by up to 1e-10 between runs), and checks
that repeated trainings agree."""

import json
from importlib.metadata import version
from typing import Any

from omai_oracle.fixture import Case, case
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

REFERENCE: dict[str, str] = {"tokenizers": version("tokenizers")}

CORPUS: list[str] = [
    "low lower lowest newer wider new news",
    "the quick brown fox jumps over the lazy dog",
    "lowly newest widest renew knew, the newer news!",
    "hello help helpful helper yellow fellow mellow",
    "unhappy unhelpful unknown unlike",
] * 2 + ["low low lower newest newest widest helpful helpful"]

HELD_OUT: list[str] = ["▁lowering", "▁newsworthy", "▁helpless", "▁xyz", "▁the", "▁unhelpful"]

CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("small", {"vocab_size": 60, "special_tokens": ["<unk>"], "unk_token": "<unk>"}),
    ("medium", {"vocab_size": 120, "special_tokens": ["<pad>", "<unk>"], "unk_token": "<unk>"}),
    ("added-unk", {"vocab_size": 80, "special_tokens": ["<s>"], "unk_token": "<unk>"}),
    ("short-pieces", {"vocab_size": 70, "max_piece_length": 3, "shrinking_factor": 0.5, "n_sub_iterations": 3}),
]


def train(cfg: dict[str, Any]) -> Tokenizer:
    """Train a Unigram tokenizer on Metaspace pre-tokens."""
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.train_from_iterator(CORPUS, trainers.UnigramTrainer(show_progress=False, **cfg))
    return tok


def canonical(tok: Tokenizer) -> list[list[Any]]:
    """The pieces with scores rounded to 1e-7, sorted by score (descending) and piece."""
    vocab = json.loads(tok.to_str())["model"]["vocab"]
    return sorted([[str(p), round(float(s), 7)] for p, s in vocab], key=lambda ps: (-ps[1], ps[0]))


def cases() -> list[Case]:
    out: list[Case] = []
    words = sorted({w for line in CORPUS for w, _ in pre_tokenizers.Metaspace().pre_tokenize_str(line)})
    for name, cfg in CONFIGS:
        tok = train(cfg)
        pieces = canonical(tok)
        for _ in range(3):
            if canonical(train(cfg)) != pieces:
                raise RuntimeError(f"unigram/{name}: HF training is not reproducible here")
        options = {
            "vocabSize": cfg["vocab_size"],
            "specialTokens": cfg.get("special_tokens", []),
            **({"unkToken": cfg["unk_token"]} if "unk_token" in cfg else {}),
            **({"maxPieceLength": cfg["max_piece_length"]} if "max_piece_length" in cfg else {}),
            **({"shrinkingFactor": cfg["shrinking_factor"]} if "shrinking_factor" in cfg else {}),
            **({"nSubIterations": cfg["n_sub_iterations"]} if "n_sub_iterations" in cfg else {}),
        }
        held = [[t.value for t in tok.model.tokenize(w)] for w in HELD_OUT]
        corpus_tokens = [[t.value for t in tok.model.tokenize(w)] for w in words]
        out.append(
            case(
                f"unigram/{name}",
                "trainUnigram",
                [CORPUS, options],
                pieces,
                rtol=0,
                atol=2e-7,
                heldOut=HELD_OUT,
                encodings=held,
                words=words,
                wordEncodings=corpus_tokens,
            )
        )
    fixed = [
        ("<unk>", 0.0),
        ("▁", -2.0),
        ("a", -3.0),
        ("b", -3.2),
        ("ab", -4.0),
        ("▁ab", -5.5),
        ("ba", -4.1),
        ("c", -6.0),
    ]
    model = models.Unigram(fixed, 0, False)
    tok = Tokenizer(model)
    for w in ["▁abab", "▁ba", "abcab", "▁xyab", "zz", "▁"]:
        out.append(
            case(
                f"tokenize/{w}",
                "unigramTokenize",
                [[list(p) for p in fixed], w],
                [[t.value, t.id, list(t.offsets)] for t in tok.model.tokenize(w)],
            )
        )
    return out
