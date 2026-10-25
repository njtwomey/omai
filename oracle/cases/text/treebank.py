"""The Penn Treebank tokeniser and detokeniser and nltk's improved word tokeniser against nltk."""

from importlib.metadata import version

from mdml_oracle.fixture import Case, case
from nltk.tokenize import NLTKWordTokenizer, TreebankWordDetokenizer, TreebankWordTokenizer

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

TEXTS: list[str] = [
    "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.",
    "They'll save and invest more.",
    "hi, my name can't hello,",
    "He said \"no\", then 'yes' -- and (maybe) [never] {ok} <x>; 50% @home #1 & co.",
    "I cannot gimme gonna gotta lemme wanna go. 'Tis 'twas d'ye more'n THEY'RE WE'VE I'M",
    "“Wait…” — he said.. ok*  ‘quoted’ «guillemets» 'em and 'tis 3,36 euros: 10:30",
    "The U.S.A. is big... isn't it?! Yes!",
    "a,\n",
    "",
]


def cases() -> list[Case]:
    out: list[Case] = []
    tb = TreebankWordTokenizer()
    nw = NLTKWordTokenizer()
    detok = TreebankWordDetokenizer()
    for i, t in enumerate(TEXTS):
        toks = tb.tokenize(t)
        out.append(case(f"treebank/{i}", "treebankTokenize", [t], toks))
        conv = tb.tokenize(t, convert_parentheses=True)
        out.append(case(f"treebank/convert/{i}", "treebankTokenize", [t], conv, convertParentheses=True))
        out.append(case(f"spans/{i}", "treebankSpans", [t], [list(s) for s in tb.span_tokenize(t)]))
        out.append(case(f"nltk/{i}", "nltkWordTokenize", [t], nw.tokenize(t)))
        out.append(case(f"detokenize/{i}", "treebankDetokenize", [toks], detok.detokenize(toks)))
        out.append(
            case(
                f"detokenize/convert/{i}",
                "treebankDetokenize",
                [conv],
                detok.detokenize(conv, convert_parentheses=True),
                convertParentheses=True,
            )
        )
    return out
