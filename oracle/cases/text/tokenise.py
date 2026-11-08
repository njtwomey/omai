"""Tokenisers against nltk (regexp_tokenize, wordpunct_tokenize), Python's str.split, scikit-learn's token pattern,
the `regex` module (\\X grapheme clusters, and the GPT-2, cl100k and o200k pre-tokeniser patterns as tiktoken writes
them) and HF tokenizers' pre-tokenisers (Whitespace, BertPreTokenizer, Split with each delimiter behaviour)."""

import re
from importlib.metadata import version
from typing import Literal

import regex
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize
from omai_oracle.fixture import Case, case
from tokenizers import pre_tokenizers

REFERENCE: dict[str, str] = {
    "nltk": version("nltk"),
    "regex": version("regex"),
    "scikit-learn": version("scikit-learn"),
    "tokenizers": version("tokenizers"),
}

TEXTS: list[str] = [
    "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.",
    "I can't stop, naïve café! It's 12345 o'clock... e.g. U.S.A.",
    "Hello world's   12345!!   tabs\there\r\nand ſome CAPS'S and THEY'LL x_y",
    "日本語のテキスト 한국어 текст ελληνικά عربى 1²³ Ⅻ",
    "é 👍🏽 🇮🇪 👨‍👩‍👧 ﬁ",
    "",
]

GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
O200K = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

Behavior = Literal["removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous"]
BEHAVIORS: list[Behavior] = ["removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous"]
CAMEL = {
    "removed": "removed",
    "isolated": "isolated",
    "merged_with_previous": "mergedWithPrevious",
    "merged_with_next": "mergedWithNext",
    "contiguous": "contiguous",
}


def spans(pre: pre_tokenizers.PreTokenizer, text: str) -> list[list[int]]:
    """The offsets (in code points) of each piece of an HF pre-tokeniser."""
    return [[a, b] for _, (a, b) in pre.pre_tokenize_str(text)]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, t in enumerate(TEXTS):
        out.append(case(f"regexTokenize/words/{i}", "regexTokenize", [t], regexp_tokenize(t, r"\w+"), pattern="\\w+"))
        out.append(
            case(f"regexTokenize/gaps/{i}", "regexTokenize", [t], regexp_tokenize(t, r"\s+", gaps=True), gaps=True)
        )
        out.append(case(f"wordPunct/{i}", "wordPunctTokenize", [t], wordpunct_tokenize(t)))
        out.append(case(f"whitespace/{i}", "whitespaceTokenize", [t], t.split()))
        out.append(case(f"sklearn/{i}", "wordTokenize", [t], re.compile(r"(?u)\b\w\w+\b").findall(t)))
        out.append(case(f"grapheme/{i}", "graphemeTokenize", [t], regex.findall(r"\X", t)))
        for name, pattern in [("gpt2", GPT2), ("cl100k", CL100K), ("o200k", O200K)]:
            out.append(case(f"preTokenize/{name}/{i}", "preTokenize", [t], regex.findall(pattern, t), pattern=name))
        out.append(case(f"hf/whitespace/{i}", "hfWhitespace", [t], spans(pre_tokenizers.Whitespace(), t)))
        out.append(case(f"hf/bert/{i}", "bertPreTokenize", [t], spans(pre_tokenizers.BertPreTokenizer(), t)))
        for b in BEHAVIORS:
            for invert in [False, True]:
                pre = pre_tokenizers.Split(" ", behavior=b, invert=invert)
                out.append(
                    case(
                        f"hf/split/{b}/{invert}/{i}",
                        "splitSpans",
                        [t],
                        spans(pre, t),
                        delimiter=" ",
                        behavior=CAMEL[b],
                        invert=invert,
                    )
                )
    return out
