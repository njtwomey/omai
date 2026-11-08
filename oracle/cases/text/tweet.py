"""nltk's TweetTokenizer and its helpers (HTML entities, handles, lengthening) against nltk."""

from importlib.metadata import version

from nltk.tokenize import TweetTokenizer
from nltk.tokenize.casual import _replace_html_entities, reduce_lengthening, remove_handles  # pyright: ignore[reportPrivateUsage]
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

TEXTS: list[str] = [
    "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--",
    "@remy: This is waaaaayyyy too much for you!!!!!!",
    "@remy Yo!! @_x @a_very_long_handle_name_here a@b.com foo@gmail.com",
    "Check https://t.co/abc(def) and www.example.com/path?q=1. Or example.org!",
    "Call me at +1 (800) 555-1234 or 555.123.4567, ok? 3/4 and -2.5+ and 10:30",
    "Price: &pound;100 &#8364;5 &#x41; &#128; &#129; &bogus; &amp;&lt;",
    "Emoji 👍🏽 👨‍👩‍👧 🇮🇪 🏴󠁧󠁢󠁳󠁣󠁴󠁿 :D 8-) </3 <b>bold</b>",
    "I'm can't won't rock-n-roll ... . . . . so...",
    "ÜBER Café NAÏVE x_y_z ##hash #tag's",
    "",
]


def cases() -> list[Case]:
    out: list[Case] = []
    configs: list[tuple[str, dict[str, bool]]] = [
        ("default", {}),
        ("lower", {"preserve_case": False}),
        ("strip", {"strip_handles": True, "reduce_len": True}),
        ("nophone", {"match_phone_numbers": False}),
    ]
    camel = {
        "preserve_case": "preserveCase",
        "strip_handles": "stripHandles",
        "reduce_len": "reduceLen",
        "match_phone_numbers": "matchPhoneNumbers",
    }
    for i, t in enumerate(TEXTS):
        for name, cfg in configs:
            tok = TweetTokenizer(**cfg)
            options = {camel[k]: v for k, v in cfg.items()}
            out.append(case(f"tweet/{name}/{i}", "tweetTokenize", [t, options], tok.tokenize(t)))
        out.append(case(f"entities/{i}", "replaceHtmlEntities", [t], _replace_html_entities(t)))
        out.append(case(f"handles/{i}", "removeHandles", [t], remove_handles(t)))
        out.append(case(f"lengthening/{i}", "reduceLengthening", [t], reduce_lengthening(t)))
    return out
