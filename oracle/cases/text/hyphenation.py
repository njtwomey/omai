"""Liang hyphenation against pyphen with its en_US dictionary: hyphenation points and hyphenated words for several
minimum fragment lengths. The fixture keeps the dictionary's patterns that can match the test words (those whose
letters occur in some `.word.`), which determine the result completely."""

import re
from importlib.metadata import version

import pyphen
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"pyphen": version("pyphen")}

TEXT: str = (
    "hyphenation algorithm computer typography dictionary representation Table supercalifragilistic "
    "encyclopedia university programming automatically information mathematics phonetic beautiful "
    "a an the strength rhythm Naïve"
)
WORDS: list[str] = [str(w) for w in TEXT.split()]


def relevant_patterns() -> list[str]:
    path = pyphen.LANGUAGES["en_US"]
    lines = path.read_text("utf-8").split("\n")[1:]
    dotted = [f".{w.lower()}." for w in WORDS]
    out: list[str] = []
    for line in lines:
        p = line.strip()
        if not p or p.startswith(("%", "#", "LEFTHYPHENMIN", "RIGHTHYPHENMIN", "COMPOUND")):
            continue
        letters = re.sub(r"\d", "", p)
        if any(letters in d for d in dotted):
            out.append(p)
    return out


def cases() -> list[Case]:
    patterns = relevant_patterns()
    out: list[Case] = []
    for left, right in [(2, 2), (2, 3), (1, 1)]:
        dic = pyphen.Pyphen(lang="en_US", left=left, right=right)
        out.append(
            case(
                f"pyphen/{left}-{right}",
                "hyphenate",
                [patterns, WORDS],
                [[list(dic.positions(w)), dic.inserted(w)] for w in WORDS],
                left=left,
                right=right,
            )
        )
    return out
