"""Tokenizer pipelines against HF tokenizers: BERT-style (BertNormalizer, BertPreTokenizer, WordPiece, template
post-processing), GPT-2 and RoBERTa-style byte-level BPE, SentencePiece-style Unigram with Metaspace, Llama-style BPE
with byte fallback and a decoder chain, added tokens with lstrip, rstrip, single-word and normalized matching, and a
WordLevel pipeline with Digits, Punctuation and Split pre-tokenisers. Each case keeps the tokenizer.json, so the test
also exercises loading; encodings keep ids, tokens, character offsets, type ids, masks, word and sequence ids, the
overflowing windows of truncation, padding, and the decoded text."""

import json
from importlib.metadata import version
from typing import Any

from tokenizers import (
    AddedToken,
    Encoding,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"tokenizers": version("tokenizers")}

CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! Hello there, world of tokenizers.",
    "unaffable affable able unable; running runner runs",
    "Café naïve résumé, crème brûlée — 日本語 テキスト 😀",
    "Numbers 12345 and 3.14159 and 2024-10-24.",
] * 3

TEXTS: list[str] = [
    "Hello world!",
    "The quick brown fox, unaffable and running.",
    "Café NAÏVE résumé 😀 日本語",
    "  leading and   multiple   spaces  ",
    "Numbers 12345!",
    "",
]
PAIRS: list[tuple[str, str]] = [("Hello world!", "The lazy dog."), ("unaffable", "Café 😀 running fast")]


def words(text: str) -> list[str]:
    """Whitespace-separated words."""
    return text.split()


def summary(e: Encoding) -> dict[str, Any]:
    return {
        "ids": e.ids,
        "tokens": e.tokens,
        "offsets": [list(o) for o in e.offsets],
        "typeIds": e.type_ids,
        "attentionMask": e.attention_mask,
        "specialTokensMask": e.special_tokens_mask,
        "wordIds": [w if w is not None else None for w in e.word_ids],
        "sequenceIds": list(e.sequence_ids),
        "overflowing": [summary(o) for o in e.overflowing],
    }


def bert() -> Tokenizer:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += words("the quick brown fox , . ! ; and over lazy dog hello world there of un aff able run cafe naive")
    vocab += words("##aff ##able ##ning ##ner ##s ##ers ##izer resume 日 本 語 numbers 1 ##2 ##3 ##4 ##5 - a")
    tok = Tokenizer(
        models.WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="[UNK]", max_input_chars_per_word=20)
    )
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tok.decoder = decoders.WordPiece()
    tok.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    return tok


def byte_bpe(roberta: bool) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=roberta)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<s>", "</s>", "<pad>"] if roberta else ["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    if roberta:
        tok.post_processor = processors.RobertaProcessing(("</s>", 1), ("<s>", 0))
    else:
        tok.post_processor = processors.ByteLevel(trim_offsets=True)
    return tok


def unigram() -> Tokenizer:
    trained = Tokenizer(models.Unigram())
    trained.pre_tokenizer = pre_tokenizers.Metaspace()
    trained.train_from_iterator(
        CORPUS,
        trainers.UnigramTrainer(
            vocab_size=150, special_tokens=["<unk>", "</s>"], unk_token="<unk>", show_progress=False
        ),
    )
    raw = json.loads(trained.to_str())["model"]["vocab"]
    # A fixed piece order: HF gives characters it had to add scores 1e-4 apart in hash order, so scores are rounded
    # to 1e-3 and ties ordered by piece.
    special = [(p, 0.0) for p, _ in raw[:2]]
    rest = sorted(((str(p), round(float(s), 3)) for p, s in raw[2:]), key=lambda ps: (-ps[1], ps[0]))
    tok = Tokenizer(models.Unigram(special + rest, 0, False))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")])
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.post_processor = processors.TemplateProcessing(
        single="$A </s>", pair="$A $B </s>", special_tokens=[("</s>", 1)]
    )
    tok.add_special_tokens(["<unk>", "</s>"])
    return tok


def llama() -> Tokenizer:
    base = Tokenizer(models.BPE())
    base.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    base.train_from_iterator(
        CORPUS, trainers.BpeTrainer(vocab_size=200, special_tokens=["<unk>", "<s>", "</s>"], show_progress=False)
    )
    d = json.loads(base.to_str())["model"]
    vocab: dict[str, int] = dict(d["vocab"])
    for b in range(256):
        vocab.setdefault(f"<0x{b:02X}>", len(vocab))
    pairs = [m if isinstance(m, list) else str(m).split(" ") for m in d["merges"]]
    merges = [(str(m[0]), str(m[1])) for m in pairs]
    tok = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", 1)]
    )
    tok.add_special_tokens(["<unk>", "<s>", "</s>"])
    return tok


def added() -> Tokenizer:
    vocab = {t: i for i, t in enumerate(words("[UNK] i read a book yesterday hey monday my phone ony"))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens([AddedToken("<mask>", lstrip=True, rstrip=False, single_word=False, normalized=False)])
    tok.add_tokens(
        [
            AddedToken("[DAY]", normalized=False),
            AddedToken("yesterday", normalized=True),
            AddedToken("ony", single_word=True, normalized=False),
            AddedToken("my", lstrip=True, rstrip=True, normalized=False),
        ]
    )
    return tok


def wordlevel() -> Tokenizer:
    names = words("[UNK] [PAD] numbers 1 2 3 4 5 12345 and . , ! - 2024 10 24 3 14159 hello world the quick")
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(dict.fromkeys(names))}, unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.Lowercase(), normalizers.NFD(), normalizers.StripAccents()]
    )
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\s+"), "removed"),
            pre_tokenizers.Punctuation("isolated"),
            pre_tokenizers.Digits(individual_digits=False),
        ]
    )
    return tok


def encodings(tok: Tokenizer) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in TEXTS:
        e = tok.encode(t)
        out.append(
            {
                "text": t,
                "encoding": summary(e),
                "decoded": tok.decode(e.ids),
                "decodedAll": tok.decode(e.ids, skip_special_tokens=False),
            }
        )
    for a, b in PAIRS:
        out.append({"text": a, "pair": b, "encoding": summary(tok.encode(a, b))})
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    builders = [
        ("bert", bert),
        ("gpt2", lambda: byte_bpe(False)),
        ("roberta", lambda: byte_bpe(True)),
        ("unigram", unigram),
        ("llama", llama),
        ("wordlevel", wordlevel),
    ]
    for name, build in builders:
        tok = build()
        out.append(case(f"{name}/plain", "encode", [tok.to_str()], encodings(tok)))
        tok.enable_truncation(max_length=12, stride=1, strategy="longest_first")
        tok.enable_padding(pad_id=0, pad_token=tok.id_to_token(0) or "[PAD]", length=14)
        out.append(case(f"{name}/truncated-padded", "encode", [tok.to_str()], encodings(tok)))
        tok.no_padding()
        tok.enable_truncation(max_length=16, stride=0, strategy="only_second", direction="left")
        tok.enable_padding(direction="left", pad_to_multiple_of=4, pad_id=0, pad_token=tok.id_to_token(0) or "[PAD]")
        pairs = [summary(e) for e in tok.encode_batch([list(p) for p in PAIRS])]
        out.append(case(f"{name}/batch", "encodeBatch", [tok.to_str(), [list(p) for p in PAIRS]], pairs))
    tok = added()
    texts = [
        "I read a book Yesterday",
        "Hey [DAY] monday",
        "my phone <mask> is phony ony",
        "a   <mask>   b",
        "Yesterday!",
    ]
    out.append(
        case(
            "added/tokens",
            "encodeTexts",
            [tok.to_str(), texts],
            [summary(tok.encode(t)) for t in texts],
        )
    )
    return out
