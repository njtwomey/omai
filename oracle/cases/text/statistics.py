"""Corpus statistics: Zipf and Heaps fits against numpy.polyfit (log-log lines) and the finite Zipf maximum-likelihood
exponent against scipy.stats.zipfian (minimising its negative log-likelihood), and topic coherence (u_mass, c_npmi)
against gensim's CoherenceModel."""

from collections import Counter
from importlib.metadata import version

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from omai_oracle.fixture import Case, case
from scipy.optimize import minimize_scalar
from scipy.stats import zipfian

REFERENCE: dict[str, str] = {"numpy": version("numpy"), "scipy": version("scipy"), "gensim": version("gensim")}

TEXT: str = (
    "the cat sat on the mat the dog sat on the log the cat saw the dog and the dog saw the cat "
    "a bird flew over the house and the cat watched the bird while the dog slept on the mat "
    "the house was quiet and the mat was warm and the cat and the dog and the bird were friends"
)
TOKENS: list[str] = [str(w) for w in TEXT.split()]

TEXTS: list[list[str]] = [
    [str(w) for w in s.split()]
    for s in [
        "human interface computer",
        "survey user computer system response time",
        "eps user interface system",
        "system human system eps",
        "user response time",
        "trees",
        "graph trees",
        "graph minors trees",
        "graph minors survey",
        "user interface response time computer system human",
    ]
]
TOPICS: list[list[str]] = [
    ["human", "computer", "system", "interface"],
    ["graph", "minors", "trees", "eps"],
    ["user", "response", "time"],
]


def cases() -> list[Case]:
    out: list[Case] = []
    counts = sorted(Counter(TOKENS).values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(counts), 1)
    out.append(case("zipf/loglog", "zipf", [TOKENS], np.array([-slope, np.exp(intercept)]), rtol=1e-10, atol=1e-12))
    n = len(counts)
    sample = np.repeat(ranks, counts)

    def nll(s: float) -> float:
        return float(-np.sum(zipfian.logpmf(sample, s, n)))

    res = minimize_scalar(nll, bounds=(1e-6, 10), method="bounded", options={"xatol": 1e-12})
    out.append(case("zipf/mle", "zipfMle", [TOKENS], np.array(float(res.x)), rtol=1e-7, atol=1e-8))
    vocab: set[str] = set()
    growth: list[int] = []
    for t in TOKENS:
        vocab.add(t)
        growth.append(len(vocab))
    beta, logk = np.polyfit(np.log(np.arange(1, len(TOKENS) + 1)), np.log(growth), 1)
    out.append(case("heaps", "heaps", [TOKENS], np.array([np.exp(logk), beta]), rtol=1e-10, atol=1e-12))
    dictionary = Dictionary(TEXTS)
    for measure in ["u_mass", "c_npmi"]:
        cm = CoherenceModel(topics=TOPICS, texts=TEXTS, dictionary=dictionary, coherence=measure)
        per = cm.get_coherence_per_topic()
        out.append(
            case(
                f"coherence/{measure}",
                "coherence",
                [TOPICS, TEXTS],
                np.array([*per, cm.get_coherence()]),
                rtol=1e-10,
                atol=1e-12,
                measure=measure,
            )
        )
    cm = CoherenceModel(topics=TOPICS, texts=TEXTS, dictionary=dictionary, coherence="c_npmi", window_size=3)
    out.append(
        case(
            "coherence/c_npmi/3",
            "coherence",
            [TOPICS, TEXTS],
            np.array([*cm.get_coherence_per_topic(), cm.get_coherence()]),
            rtol=1e-10,
            atol=1e-12,
            measure="c_npmi",
            window=3,
        )
    )
    return out
