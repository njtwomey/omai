"""Soundex and Metaphone against jellyfish."""

from importlib.metadata import version

import jellyfish
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"jellyfish": version("jellyfish")}

TEXT: str = """
Robert Rupert Rubin Ashcraft Ashcroft Tymczak Pfister Honeyman Lee Gutierrez Jackson Washington Thompson knight
gnome pneumonia wright aesthetic Schmidt Xavier Xhosa xiong phonetic thumb dumb bomb chair character science
ciao judge edge sign signed ghost night laugh hough who whale yellow sky quick fax tough biscuit cherry Sean
Zachary mb b Müller Ñandú ß Straße Catherine Kathryn Smith Smyth O'Brien McDonald van der Berg
"""
WORDS: list[str] = [str(w) for w in TEXT.split()]


def cases() -> list[Case]:
    return [
        case("soundex", "soundex", [WORDS], [jellyfish.soundex(w) for w in WORDS]),
        case("metaphone", "metaphone", [WORDS], [jellyfish.metaphone(w) for w in WORDS]),
        case(
            "metaphone/phrase",
            "metaphone",
            [["van der Berg", "the quick fox"]],
            [jellyfish.metaphone("van der Berg"), jellyfish.metaphone("the quick fox")],
        ),
    ]
