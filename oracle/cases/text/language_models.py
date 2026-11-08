"""N-gram smoothers against a direct implementation of Chen and Goodman's (1998) definitions, written recursively over
tuples of words with Counters: interpolated absolute discounting, Witten-Bell, interpolated Kneser-Ney (continuation
counts below the top order) and modified Kneser-Ney (three discounts from the counts of counts), at orders 2 and 3,
for seen and unseen words and contexts. NLTK's `nltk.lm` models differ in conventions (its Kneser-Ney keeps raw counts
and a uniform unigram, its Witten-Bell weights by the total count of the order), so it is not used as the reference."""

from collections import Counter

from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"python": "stdlib"}

CORPUS: list[list[str]] = [
    s.split()
    for s in [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat saw a dog",
        "the dog saw the cat",
        "a dog sat on a mat",
        "the cat ran away",
        "the dog ran home",
        "a bird sat on the log",
        "the cat sat on the log",
        "a cat ran home",
    ]
]
BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def counts(order: int) -> list[Counter[tuple[str, ...]]]:
    tables: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(order)]
    for s in CORPUS:
        p = [BOS] * (order - 1) + s + [EOS]
        for i in range(order - 1, len(p)):
            for m in range(1, order + 1):
                tables[m - 1][tuple(p[i - m + 1 : i + 1])] += 1
    return tables


def discounts(values: list[int], modified: bool) -> tuple[float, float, float]:
    nr = Counter(values)
    n1, n2, n3, n4 = nr[1], nr[2], nr[3], nr[4]
    y = n1 / (n1 + 2 * n2)
    if not modified:
        return (y, y, y)
    d1 = 1 - 2 * y * n2 / n1
    d2 = 2 - 3 * y * n3 / n2 if n3 else d1
    d3 = 3 - 4 * y * n4 / n3 if n3 and n4 else d2
    return (d1, d2, d3)


class Model:
    def __init__(self, order: int, kind: str, unigram: str) -> None:
        self.order, self.kind, self.unigram = order, kind, unigram
        raw = counts(order)
        kn = kind in ("kneser-ney", "modified-kneser-ney")
        self.level: list[Counter[tuple[str, ...]]] = []
        for m in range(1, order + 1):
            if not kn or m == order:
                self.level.append(raw[m - 1])
                continue
            c: Counter[tuple[str, ...]] = Counter()
            for g, n in raw[m - 1].items():
                if g[0] == BOS:
                    c[g] = n
                else:
                    c[g] = len({h[0] for h in raw[m] if h[1:] == g})
            self.level.append(c)
        self.words = sorted({g[0] for g in raw[0]} | {EOS, UNK})
        self.disc = [discounts(list(t.values()), kind == "modified-kneser-ney") for t in self.level]

    def context(self, m: int, h: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [g for g in self.level[m - 1] if g[:-1] == h]

    def p(self, w: str, h: tuple[str, ...], m: int | None = None) -> float:
        m = self.order if m is None else m
        h = h[len(h) - (m - 1) :] if m > 1 else ()
        grams = self.context(m, h)
        total = sum(self.level[m - 1][g] for g in grams)
        if m == 1:
            n = self.level[0][(w,)]
            if self.kind == "witten-bell":
                t = len(grams)
                return n / total if self.unigram == "mle" else (n + t / len(self.words)) / (total + t)
            if self.unigram == "mle":
                return n / total
            d = self.disc[0]
            dn = 0.0 if n == 0 else d[0] if n == 1 else d[1] if n == 2 else d[2]
            gamma = (
                (
                    d[0] * sum(1 for g in grams if self.level[0][g] == 1)
                    + d[1] * sum(1 for g in grams if self.level[0][g] == 2)
                    + d[2] * sum(1 for g in grams if self.level[0][g] >= 3)
                )
                if self.kind == "modified-kneser-ney"
                else d[0] * len(grams)
            )
            return (n - dn) / total + gamma / total / len(self.words)
        if not grams:
            return self.p(w, h, m - 1)
        lower = self.p(w, h, m - 1)
        n = self.level[m - 1][h + (w,)]
        if self.kind == "witten-bell":
            t = len(grams)
            return (n + t * lower) / (total + t)
        d = self.disc[m - 1]
        dn = 0.0 if n == 0 else d[0] if n == 1 else d[1] if n == 2 else d[2]
        vals = [self.level[m - 1][g] for g in grams]
        gamma = (
            d[0] * sum(1 for v in vals if v == 1)
            + d[1] * sum(1 for v in vals if v == 2)
            + d[2] * sum(1 for v in vals if v >= 3)
            if self.kind == "modified-kneser-ney"
            else d[0] * len(vals)
        )
        return (n - dn) / total + gamma / total * lower


QUERIES: list[tuple[str, list[str]]] = [
    ("cat", ["the"]),
    ("sat", ["the", "cat"]),
    ("dog", ["a"]),
    ("log", ["on", "the"]),
    ("</s>", ["ran", "home"]),
    ("mat", ["the", "dog"]),
    ("bird", ["the"]),
    ("zebra", ["a", "cat"]),
    ("home", ["zebra"]),
    ("cat", ["saw", "the"]),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for kind in ["absolute-discounting", "witten-bell", "kneser-ney", "modified-kneser-ney"]:
        for order in (2, 3):
            for unigram in ("uniform", "mle"):
                mdl = Model(order, kind, unigram)
                probs: list[float] = []
                for w, h in QUERIES:
                    word = w if w in mdl.words else UNK
                    ctx = tuple(x if x in mdl.words else UNK for x in h)
                    ctx = (BOS,) * (order - 1 - len(ctx)) + ctx[-(order - 1) :]
                    if unigram == "mle" and word == UNK:
                        probs.append(0.0 if kind != "witten-bell" else mdl.p(word, ctx))
                    else:
                        probs.append(mdl.p(word, ctx))
                out.append(
                    case(
                        f"{kind}/{order}/{unigram}",
                        "ngram",
                        [CORPUS, [[w, h] for w, h in QUERIES]],
                        probs,
                        rtol=1e-12,
                        atol=1e-14,
                        order=order,
                        smoothing=kind,
                        unigram=unigram,
                    )
                )
    return out
