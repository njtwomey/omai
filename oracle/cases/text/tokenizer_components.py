"""Tokenizer pipeline components on their own against HF tokenizers: each normaliser's normalize_str, each
pre-tokeniser's pre_tokenize_str (pieces with code point offsets), and each decoder's decode. Each component is passed
inside the tokenizer.json HF serialises around it, so the test also checks loading."""

import json
from importlib.metadata import version

from mdml_oracle.fixture import Case, case
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

REFERENCE: dict[str, str] = {"tokenizers": version("tokenizers")}

TEXTS: list[str] = [
    "  Héllo, WORLD!  It's 2024-10-24\tok yes​ ",
    "日本語のテキスト and Ελληνικά, 中文!",
    "áb ç \x07bell� ë",
    "x" * 3 + "▁already▁spaced",
]


def component_json(kind: str, component: object) -> str:
    """The tokenizer.json of a tokenizer that holds the component."""
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    setattr(tok, kind, component)
    return json.dumps(json.loads(tok.to_str()), ensure_ascii=False)


NORMALIZERS: list[tuple[str, normalizers.Normalizer]] = [
    ("nfkc", normalizers.NFKC()),
    ("nfd", normalizers.NFD()),
    ("lowercase", normalizers.Lowercase()),
    ("strip-accents", normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])),
    ("strip", normalizers.Strip()),
    ("strip-left", normalizers.Strip(left=True, right=False)),
    ("replace", normalizers.Replace(Regex(r"\s+"), " ")),
    ("replace-string", normalizers.Replace("o", "0")),
    ("prepend", normalizers.Prepend("▁")),
    ("bert", normalizers.BertNormalizer()),
    ("bert-cased", normalizers.BertNormalizer(lowercase=False, strip_accents=False, handle_chinese_chars=False)),
    ("nmt", normalizers.Nmt()),
    ("byte-level", normalizers.ByteLevel()),
]

PRE_TOKENIZERS: list[tuple[str, pre_tokenizers.PreTokenizer]] = [
    ("whitespace", pre_tokenizers.Whitespace()),
    ("whitespace-split", pre_tokenizers.WhitespaceSplit()),
    ("bert", pre_tokenizers.BertPreTokenizer()),
    ("byte-level", pre_tokenizers.ByteLevel(add_prefix_space=True)),
    ("byte-level-plain", pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)),
    ("metaspace", pre_tokenizers.Metaspace()),
    ("metaspace-first", pre_tokenizers.Metaspace(prepend_scheme="first")),
    ("metaspace-never", pre_tokenizers.Metaspace(prepend_scheme="never", split=False)),
    ("punctuation", pre_tokenizers.Punctuation()),
    ("punctuation-next", pre_tokenizers.Punctuation(behavior="merged_with_next")),
    ("digits", pre_tokenizers.Digits(individual_digits=True)),
    ("digits-contiguous", pre_tokenizers.Digits(individual_digits=False)),
    ("split", pre_tokenizers.Split(Regex(r"[,!]"), behavior="isolated")),
    ("split-invert", pre_tokenizers.Split(Regex(r"\w+"), behavior="removed", invert=True)),
    ("char-delimiter", pre_tokenizers.CharDelimiterSplit(" ")),
    ("fixed-length", pre_tokenizers.FixedLength(length=4)),
    (
        "sequence",
        pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Digits(individual_digits=True)]),
    ),
]

DECODERS: list[tuple[str, decoders.Decoder, list[str]]] = [
    ("byte-level", decoders.ByteLevel(), ["Hello", "Ġw", "orld", "Ġ", "ðŁĺ", "Ģ"]),
    ("wordpiece", decoders.WordPiece(), ["un", "##aff", "##able", "do", "n't", ",", "ok", "."]),
    ("wordpiece-raw", decoders.WordPiece(prefix="##", cleanup=False), ["un", "##aff", "it", "'s", "."]),
    ("metaspace", decoders.Metaspace(), ["▁Hello", "▁wor", "ld", "▁"]),
    ("metaspace-never", decoders.Metaspace(prepend_scheme="never"), ["▁Hello", "▁world"]),
    ("bpe", decoders.BPEDecoder(), ["hel", "lo</w>", "wor", "ld</w>"]),
    ("replace", decoders.Replace("▁", " "), ["▁a", "b▁▁c"]),
    ("strip", decoders.Strip(" ", 1, 1), [" a ", "  b  ", "c"]),
    ("fuse", decoders.Fuse(), ["a", "b", "c"]),
    ("byte-fallback", decoders.ByteFallback(), ["a", "<0xE2>", "<0x82>", "<0xAC>", "b", "<0xFF>", "c"]),
    ("ctc", decoders.CTC(), ["<pad>", "h", "h", "e", "<pad>", "l", "l", "<pad>", "l", "o", "|", "w", "o", "."]),
    (
        "sequence",
        decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
        ),
        ["▁Hi", "▁<0xE2>", "<0x82>", "<0xAC>", "!"],
    ),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, n in NORMALIZERS:
        spec = component_json("normalizer", n)
        out.append(case(f"normalizer/{name}", "normalizer", [spec, TEXTS], [n.normalize_str(t) for t in TEXTS]))
    for name, p in PRE_TOKENIZERS:
        spec = component_json("pre_tokenizer", p)
        pieces = [[[s, list(o)] for s, o in p.pre_tokenize_str(t)] for t in TEXTS]
        out.append(case(f"pre-tokenizer/{name}", "preTokenizer", [spec, TEXTS], pieces))
    for name, d, tokens in DECODERS:
        spec = component_json("decoder", d)
        out.append(case(f"decoder/{name}", "decoder", [spec, tokens], d.decode(tokens)))
    return out
