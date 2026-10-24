"""Unicode normalisation, case folding and accent stripping against Python's unicodedata and str.casefold,
scikit-learn's strip_accents_unicode and strip_accents_ascii, and HF tokenizers' StripAccents (after NFD)."""

import unicodedata
from importlib.metadata import version
from typing import Literal

from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode
from tokenizers import normalizers
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "tokenizers": version("tokenizers")}

TEXTS: list[str] = [
    "Straße",
    "ΌΣΟΣ όσος",
    "café naïve résumé",
    "é ä ǫ́",
    "ﬁﬂ ① ½ Ω Å",
    "Ǆ ǅ ǆ İ ı",
    "Ꮳ ꮳ ᏸ",
    "Ångström Øre Æsir",
    "日本語 テキスト ｶﾀｶﾅ",
    "Crème BRÛLÉE",
    "",
]


def strip_bert(text: str) -> str:
    """HF tokenizers' accent stripping: NFD, then remove non-spacing marks."""
    return normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()]).normalize_str(text)


def cases() -> list[Case]:
    out: list[Case] = []
    for i, t in enumerate(TEXTS):
        forms: list[Literal["NFC", "NFD", "NFKC", "NFKD"]] = ["NFC", "NFD", "NFKC", "NFKD"]
        for form in forms:
            out.append(
                case(f"normalize/{form}/{i}", "normalizeUnicode", [t], unicodedata.normalize(form, t), form=form)
            )
        out.append(case(f"caseFold/{i}", "caseFold", [t], t.casefold()))
        out.append(case(f"stripAccents/unicode/{i}", "stripAccents", [t], strip_accents_unicode(t), mode="unicode"))
        out.append(case(f"stripAccents/ascii/{i}", "stripAccents", [t], strip_accents_ascii(t), mode="ascii"))
        out.append(case(f"stripAccents/nonspacing/{i}", "stripAccents", [t], strip_bert(t), mode="nonspacing"))
    return out
