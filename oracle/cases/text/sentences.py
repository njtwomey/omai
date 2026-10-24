"""Punkt sentence segmentation against nltk: training (abbreviations, collocations, sentence starters, orthographic
context) on a short text, and sentence spans with the trained parameters, with none, and with a given abbreviation
list."""

from importlib.metadata import version

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer, PunktTrainer
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

TRAIN: str = (
    "Mr. Smith went to Washington on Jan. 5 of this year. He met Dr. Jones there. "
    "The U.S. government was closed, e.g. on weekends. Mrs. Brown, however, was not amused.\n"
    "They visited St. Louis and then flew to Los Angeles. It was 3 p.m. when they landed. "
    "J. R. R. Tolkien wrote books. Prof. Adams teaches at the univ. in town. "
    "Mr. Smith and Dr. Jones agreed. The meeting ended at 5 p.m. Everyone left.\n\n"
    "In 1999. the company grew. Sales rose 5.2 per cent. in Q3. A new CEO, Mr. Lee, arrived. "
    'Mr. Lee said: "We will grow." Then he smiled. Dr. Jones replied (quietly). '
    "The No. 1 priority is safety. Fig. 3 shows the results. See vol. 2, p. 14 for details. "
    "He arrived at approx. 9 o'clock. They wanted cheap and fast service... It never came. "
    "Mrs. Brown wrote to Mr. Smith. Dr. Jones wrote to Mrs. Brown. "
) * 3

TESTS: list[str] = [
    "Mr. Smith met Dr. Jones at 5 p.m. on Jan. 5. They talked.",
    'He said "Hello." Then he left. (Really?) Yes!',
    "The U.S. economy grew 3.5 per cent. in Q2. J. R. R. Tolkien agreed... Others did not.",
    "No breaks here",
    "Ends with spaces.   ",
    "First line.\n\nNew paragraph starts here. And continues.",
    "",
]


def cases() -> list[Case]:
    trainer = PunktTrainer()
    trainer.train(TRAIN, finalize=True)
    params = trainer.get_params()
    learned = {
        "abbrevTypes": sorted(params.abbrev_types),
        "collocations": sorted(f"{a} {b}" for a, b in params.collocations),
        "sentStarters": sorted(params.sent_starters),
        "orthoContext": sorted([k, v] for k, v in params.ortho_context.items()),
    }
    out = [case("train", "trainPunkt", [TRAIN], learned)]
    trained = PunktSentenceTokenizer(params)
    plain = PunktSentenceTokenizer()
    custom = PunktParameters()
    custom.abbrev_types = {"mr", "dr", "jan", "p.m", "u.s"}
    ruled = PunktSentenceTokenizer(custom)
    for i, t in enumerate([*TESTS, TRAIN[:600]]):
        out.append(
            case(f"trained/{i}", "sentenceSpans", [t], [list(s) for s in trained.span_tokenize(t)], params="trained")
        )
        out.append(case(f"plain/{i}", "sentenceSpans", [t], [list(s) for s in plain.span_tokenize(t)], params="plain"))
        out.append(case(f"rules/{i}", "sentenceSpans", [t], [list(s) for s in ruled.span_tokenize(t)], params="rules"))
        out.append(
            case(
                f"unaligned/{i}",
                "sentenceSpans",
                [t],
                [list(s) for s in trained.span_tokenize(t, realign_boundaries=False)],
                params="trained",
                realign=False,
            )
        )
    return out
