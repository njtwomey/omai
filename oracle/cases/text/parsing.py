"""Context-free and probabilistic parsing against nltk: every parse of ambiguous sentences (ChartParser and
EarleyChartParser) as bracketed trees, the Viterbi parse and its probability (ViterbiParser), and the sentence
probability as the sum over InsideChartParser's parses."""

from importlib.metadata import version

from mdml_oracle.fixture import Case, case
from nltk import CFG, PCFG
from nltk.parse import ChartParser, EarleyChartParser, ViterbiParser
from nltk.parse.pchart import InsideChartParser
from nltk.tree import Tree

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

GRAMMAR: str = """
S -> NP VP
NP -> Det N | NP PP | 'I' | 'John'
VP -> V NP | VP PP | V
PP -> P NP
Det -> 'the' | 'a'
N -> 'man' | 'telescope' | 'dog' | 'park'
V -> 'saw' | 'walked'
P -> 'with' | 'in'
"""

PROB: str = """
S -> NP VP [1.0]
NP -> Det N [0.5] | NP PP [0.2] | 'I' [0.2] | 'John' [0.1]
VP -> V NP [0.6] | VP PP [0.3] | V [0.1]
PP -> P NP [1.0]
Det -> 'the' [0.7] | 'a' [0.3]
N -> 'man' [0.3] | 'telescope' [0.2] | 'dog' [0.3] | 'park' [0.2]
V -> 'saw' [0.6] | 'walked' [0.4]
P -> 'with' [0.5] | 'in' [0.5]
"""

SENTENCES: list[str] = [
    "I saw the man with the telescope",
    "John walked the dog in the park with a telescope",
    "I walked",
    "the dog saw John",
    "saw the man",
]


def flat(t: Tree) -> str:
    return t.pformat(margin=1_000_000)


def cases() -> list[Case]:
    out: list[Case] = []
    cfg = CFG.fromstring(GRAMMAR)
    pcfg = PCFG.fromstring(PROB)
    for i, s in enumerate(SENTENCES):
        toks = s.split()
        chart = sorted(flat(t) for t in ChartParser(cfg).parse(toks))
        earley = sorted(flat(t) for t in EarleyChartParser(cfg).parse(toks))
        out.append(case(f"cfg/{i}", "parses", [GRAMMAR, toks], chart, earley=earley))
        best = list(ViterbiParser(pcfg).parse(toks))
        total = sum(t.prob() for t in InsideChartParser(pcfg).parse(toks))
        out.append(
            case(
                f"pcfg/{i}",
                "pcfg",
                [PROB, toks],
                [flat(best[0]) if best else "", best[0].prob() if best else 0.0, total],
                rtol=1e-12,
                atol=1e-15,
            )
        )
    return out
