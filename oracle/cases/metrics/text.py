"""Text metrics: edit rates from jiwer, BLEU, chrF and TER from sacrebleu, ROUGE from rouge-score, SQuAD exact match
and F1, BERTScore from embeddings and perplexity from torchmetrics; distinct-n from its definition.
"""

from collections import Counter
from importlib.metadata import version
from typing import Any

import jiwer
import numpy as np
import sacrebleu
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from torchmetrics.functional.text import perplexity
from torchmetrics.functional.text.bert import _get_precision_recall_f1  # pyright: ignore[reportPrivateUsage]
from torchmetrics.functional.text.squad import (
    _compute_exact_match_score,  # pyright: ignore[reportPrivateUsage]
    _compute_f1_score,  # pyright: ignore[reportPrivateUsage]
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "jiwer": version("jiwer"),
    "sacrebleu": version("sacrebleu"),
    "rouge-score": version("rouge-score"),
    "torchmetrics": version("torchmetrics"),
}

REFS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "It cost $3.50, or 4-5 euros, at the time!",
    "A man, a plan, a canal: Panama.",
    "The cat sat on the mat and looked at the dog.",
    "We're going to the U.S. next year (probably).",
    "Data-driven methods often outperform hand-crafted rules.",
]
HYPS: list[str] = [
    "the quick brown fox jumped over a lazy dog",
    "It cost 3.50 dollars, or 4-5 euro at that time.",
    "A plan, a man, a canal Panama!",
    "On the mat the cat sat and looked at a dog.",
    "We are going to the US next year probably.",
    "Hand-crafted rules are often outperformed by data-driven methods.",
]
REFS2: list[str] = [
    "A quick brown fox leaps over the lazy dog.",
    "Back then it cost $3.50 or 4-5 euros!",
    "A man a plan a canal Panama",
    "The cat was sitting on the mat looking at the dog.",
    "Next year we will probably go to the U.S.",
    "Data-driven approaches often beat hand-crafted rules.",
]
SUMMARY_REF = "the cat sat on the mat.\nthe dog lay by the door.\nit was a quiet afternoon."
SUMMARY_HYP = "a quiet afternoon.\nthe cat lay on the mat\nand the dog sat by the door."


def random_sentences(seed: int, n: int) -> tuple[list[str], list[str]]:
    g = rng(seed)
    vocab = ["a", "b", "c", "d", "e", "f"]
    refs: list[str] = []
    hyps: list[str] = []
    for _ in range(n):
        refs.append(" ".join(vocab[int(i)] for i in g.integers(0, 6, size=int(g.integers(1, 14)))))
        hyps.append(" ".join(vocab[int(i)] for i in g.integers(0, 6, size=int(g.integers(1, 14)))))
    return refs, hyps


def edit_cases() -> list[Case]:
    rr, rh = random_sentences(80, 60)
    out: list[Case] = []
    for tag, refs, hyps in [("prose", REFS, HYPS), ("random", rr, rh)]:
        words = jiwer.process_words(refs, hyps)
        chars = jiwer.process_characters(refs, hyps)
        for name, value in [
            ("wordErrorRate", words.wer),
            ("matchErrorRate", words.mer),
            ("wordInformationPreserved", words.wip),
            ("wordInformationLost", words.wil),
            ("characterErrorRate", chars.cer),
        ]:
            out.append(case(f"{name}/{tag}", name, [], np.asarray(float(value)), yTrue=refs, yPred=hyps))
        out.append(
            case(
                f"wordErrorRate/{tag}/counts",
                "editCounts",
                [],
                np.asarray([words.hits, words.substitutions, words.deletions, words.insertions], dtype=float),
                yTrue=refs,
                yPred=hyps,
            )
        )
    return out


def bleu_cases() -> list[Case]:
    out: list[Case] = []
    both = [[a, b] for a, b in zip(REFS, REFS2, strict=True)]
    for smooth in ["exp", "floor", "add-k", "none"]:
        score = sacrebleu.corpus_bleu(HYPS, [REFS, REFS2], smooth_method=smooth).score
        out.append(
            case(f"bleuScore/{smooth}", "bleuScore", [], np.asarray(score), yTrue=both, yPred=HYPS, smooth=smooth)
        )
    one = sacrebleu.corpus_bleu(HYPS, [REFS], lowercase=True, tokenize="none").score
    out.append(
        case(
            "bleuScore/lower-none",
            "bleuScore",
            [],
            np.asarray(one),
            yTrue=REFS,
            yPred=HYPS,
            lowercase=True,
            tokenize="none",
        )
    )
    for i, (h, r) in enumerate(zip(HYPS, REFS, strict=True)):
        s = sacrebleu.sentence_bleu(h, [r, REFS2[i]]).score
        out.append(case(f"sentenceBleu/{i}", "sentenceBleu", [], np.asarray(s), yTrue=[r, REFS2[i]], yPred=h))
    return out


def chrf_cases() -> list[Case]:
    out: list[Case] = []
    both = [[a, b] for a, b in zip(REFS, REFS2, strict=True)]
    settings: list[tuple[str, dict[str, Any]]] = [
        ("chrf", {}),
        ("chrf++", {"word_order": 2}),
        ("beta1-lower", {"beta": 1, "lowercase": True}),
        ("whitespace", {"whitespace": True}),
        ("eps", {"eps_smoothing": True, "word_order": 2}),
    ]
    camel = {"word_order": "wordOrder", "eps_smoothing": "epsSmoothing"}
    for tag, kw in settings:
        score = CHRF(**kw).corpus_score(HYPS, [REFS, REFS2]).score
        params = {camel.get(k, k): v for k, v in kw.items()}
        out.append(case(f"chrfScore/{tag}", "chrfScore", [], np.asarray(score), yTrue=both, yPred=HYPS, **params))
    return out


def ter_cases() -> list[Case]:
    out: list[Case] = []
    both = [[a, b] for a, b in zip(REFS, REFS2, strict=True)]
    settings: list[tuple[str, dict[str, Any]]] = [
        ("default", {}),
        ("case", {"case_sensitive": True}),
        ("normalized", {"normalized": True}),
        ("nopunct", {"no_punct": True}),
    ]
    camel = {"case_sensitive": "caseSensitive", "no_punct": "noPunct"}
    for tag, kw in settings:
        score = TER(**kw).corpus_score(HYPS, [REFS, REFS2]).score
        params = {camel.get(k, k): v for k, v in kw.items()}
        out.append(
            case(
                f"translationEditRate/{tag}",
                "translationEditRate",
                [],
                np.asarray(score),
                yTrue=both,
                yPred=HYPS,
                **params,
            )
        )
    rr, rh = random_sentences(81, 40)
    score = sacrebleu.corpus_ter(rh, [rr]).score
    out.append(case("translationEditRate/random", "translationEditRate", [], np.asarray(score), yTrue=rr, yPred=rh))
    return out


def rouge_cases() -> list[Case]:
    out: list[Case] = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3", "rougeL", "rougeLsum"])
    refs = [*REFS, SUMMARY_REF]
    hyps = [*HYPS, SUMMARY_HYP]
    both = [[a, b] for a, b in zip(REFS, REFS2, strict=True)] + [[SUMMARY_REF]]
    scores = [scorer.score_multi(r, h) for r, h in zip(both, hyps, strict=True)]
    for kind, fn, extra in [
        ("rouge1", "rougeN", {"n": 1}),
        ("rouge2", "rougeN", {"n": 2}),
        ("rouge3", "rougeN", {"n": 3}),
        ("rougeL", "rougeL", {}),
        ("rougeLsum", "rougeLsum", {}),
    ]:
        for field in ["precision", "recall", "fmeasure"]:
            value = float(np.mean([getattr(s[kind], field) for s in scores]))
            out.append(case(f"{kind}/{field}", fn, [], np.asarray(value), yTrue=both, yPred=hyps, field=field, **extra))
    del refs
    return out


def squad_cases() -> list[Case]:
    truths = [["Denver Broncos", "The Denver Broncos"], ["Santa Clara, California"], ["the 50th"], [""], ["gold"]]
    preds = ["denver broncos!", "Santa Clara", "50th Super Bowl", "", "golden"]
    em = 100 * np.mean(
        [max(float(_compute_exact_match_score(p, t)) for t in ts) for p, ts in zip(preds, truths, strict=True)]
    )
    f1 = 100 * np.mean([max(float(_compute_f1_score(p, t)) for t in ts) for p, ts in zip(preds, truths, strict=True)])
    return [
        case("exactMatch", "exactMatch", [], np.asarray(float(em)), yTrue=truths, yPred=preds),
        case("tokenF1", "tokenF1", [], np.asarray(float(f1)), yTrue=truths, yPred=preds, rtol=1e-6),  # float32 F1,
    ]


def bert_cases() -> list[Case]:
    g = rng(82)
    targets = [g.normal(size=(int(n), 6)) for n in [5, 3, 7]]
    preds = [g.normal(size=(int(n), 6)) for n in [4, 6, 2]]
    tw = [g.uniform(0.5, 2.0, size=len(t)) for t in targets]
    pw = [g.uniform(0.5, 2.0, size=len(p)) for p in preds]
    out: list[Case] = []
    for weighted in [False, True]:
        ps: list[float] = []
        rs: list[float] = []
        fs: list[float] = []
        for i, (t, p) in enumerate(zip(targets, preds, strict=True)):
            te = torch.tensor(t / np.linalg.norm(t, axis=1, keepdims=True))[None, None]
            pe = torch.tensor(p / np.linalg.norm(p, axis=1, keepdims=True))[None, None]
            wt = tw[i] if weighted else np.ones(len(t))
            wp = pw[i] if weighted else np.ones(len(p))
            prec, rec, f = _get_precision_recall_f1(
                pe, te, torch.tensor(wp / wp.sum())[None], torch.tensor(wt / wt.sum())[None]
            )
            ps.append(float(prec))
            rs.append(float(rec))
            fs.append(float(f))
        params: dict[str, Any] = {"targets": [t.tolist() for t in targets], "preds": [p.tolist() for p in preds]}
        if weighted:
            params["targetWeights"] = [w.tolist() for w in tw]
            params["predWeights"] = [w.tolist() for w in pw]
        tag = "idf" if weighted else "uniform"
        for field, values in [("precision", ps), ("recall", rs), ("f1", fs)]:
            out.append(
                case(
                    f"bertScore/{tag}/{field}",
                    "bertScore",
                    [],
                    np.asarray(float(np.mean(values))),
                    field=field,
                    **params,
                )
            )
    return out


def perplexity_cases() -> list[Case]:
    g = rng(83)
    logits: FloatArray = g.normal(size=(3, 5, 7)) * 2
    target = g.integers(0, 7, size=(3, 5))
    target[0, 4] = -100
    target[2, 0] = -100
    full = float(perplexity(torch.tensor(logits), torch.tensor(np.where(target < 0, 1, target))))
    masked = float(perplexity(torch.tensor(logits), torch.tensor(target), ignore_index=-100))
    return [
        case("perplexity", "perplexity", [logits, np.where(target < 0, 1, target).astype(float)], np.asarray(full)),
        case("perplexity/ignore", "perplexity", [logits, target.astype(float)], np.asarray(masked), ignoreIndex=-100),
    ]


def distinct(texts: list[str], n: int) -> float:
    grams: Counter[tuple[str, ...]] = Counter()
    for t in texts:
        tokens = t.split()
        grams.update(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return len(grams) / sum(grams.values())


def cases() -> list[Case]:
    out = edit_cases() + bleu_cases() + chrf_cases() + ter_cases() + rouge_cases() + squad_cases()
    out += bert_cases() + perplexity_cases()
    for n in [1, 2, 3]:
        out.append(case(f"distinctN/{n}", "distinctN", [], np.asarray(distinct(HYPS, n)), yPred=HYPS, n=n))
    tok = Tokenizer13a()
    lines = [*REFS, "Tom &amp; Jerry's &quot;show&quot; &lt;3 -- 1,000.5 items, e.g. x-ray.", "a-\nb <skipped> c\nd"]
    for i, line in enumerate(lines):
        out.append(case(f"tokenize13a/{i}", "tokenize13a", [], str(tok(line)), line=line))
    return out
