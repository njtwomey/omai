"""Reference implementations for the ILP fixtures: Plotkin's anti-unification and relative least general
generalisation over string terms, brute-force version spaces, and a typed port of FOIL's covering loop with Quinlan's
gain (candidate literals generated in the same documented order as the engine, so learned clauses can be compared)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

# ------------------------------------------------------------------------------------------------ terms

type T = str | tuple[str, tuple["T", ...]]
"""A term: a constant or variable name, or (functor, args)."""


def show(t: T) -> str:
    """Prolog text without spaces."""
    if isinstance(t, str):
        return t
    return f"{t[0]}({','.join(show(a) for a in t[1])})"


def parse(s: str) -> T:
    """A term from text like f(a,g(X))."""
    s = s.replace(" ", "")
    pos = 0

    def term() -> T:
        nonlocal pos
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] == "_"):
            pos += 1
        name = s[start:pos]
        if pos < len(s) and s[pos] == "(":
            pos += 1
            args = [term()]
            while s[pos] == ",":
                pos += 1
                args.append(term())
            pos += 1
            return (name, tuple(args))
        return name

    return term()


class Lgg:
    """Anti-unification with a shared table of differing pairs."""

    def __init__(self) -> None:
        self.table: dict[tuple[str, str], str] = {}

    def of(self, a: T, b: T) -> T:
        """The lgg of two terms."""
        if a == b:
            return a
        if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a[1]) == len(b[1]):
            return (a[0], tuple(self.of(x, y) for x, y in zip(a[1], b[1], strict=True)))
        key = (show(a), show(b))
        if key not in self.table:
            self.table[key] = f"G{len(self.table)}"
        return self.table[key]


def pred(t: T) -> tuple[str, int]:
    """Predicate indicator."""
    return (t, 0) if isinstance(t, str) else (t[0], len(t[1]))


def lgg_clauses(c: tuple[T, list[T]], d: tuple[T, list[T]]) -> tuple[T, list[T]]:
    """Plotkin's lgg of two clauses (ground or with distinct variables)."""
    table = Lgg()
    head = table.of(c[0], d[0])
    body: list[T] = []
    for x in c[1]:
        for y in d[1]:
            if pred(x) == pred(y):
                g = table.of(x, y)
                if g not in body:
                    body.append(g)
    return head, body


def variables(t: T) -> list[str]:
    """Variables (names starting upper case) in order."""
    if isinstance(t, str):
        return [t] if t[:1].isupper() else []
    out: list[str] = []
    for a in t[1]:
        for v in variables(a):
            if v not in out:
                out.append(v)
    return out


def connected(head: T, body: list[T]) -> list[T]:
    """Body literals linked to the head by shared variables."""
    linked = set(variables(head))
    keep: list[int] = []
    changed = True
    while changed:
        changed = False
        for i, b in enumerate(body):
            if i in keep:
                continue
            vs = variables(b)
            if any(v in linked for v in vs):
                keep.append(i)
                linked |= set(vs)
                changed = True
    return [b for i, b in enumerate(body) if i in keep]


def match(p: T, t: T, s: dict[str, T]) -> dict[str, T] | None:
    """One-way matching of p onto t (t's variables are constants)."""
    if isinstance(p, str) and p[:1].isupper():
        if p in s:
            return s if s[p] == t else None
        return {**s, p: t}
    if isinstance(p, str) or isinstance(t, str):
        return s if p == t else None
    if p[0] != t[0] or len(p[1]) != len(t[1]):
        return None
    for x, y in zip(p[1], t[1], strict=True):
        r = match(x, y, s)
        if r is None:
            return None
        s = r
    return s


def subsumes(c: tuple[T, list[T]], d: tuple[T, list[T]]) -> bool:
    """Theta-subsumption by backtracking."""
    s0 = match(c[0], d[0], {})
    if s0 is None:
        return False

    def go(i: int, s: dict[str, T]) -> bool:
        if i == len(c[1]):
            return True
        for lit in d[1]:
            r = match(c[1][i], lit, s)
            if r is not None and go(i + 1, r):
                return True
        return False

    return go(0, s0)


def reduce(c: tuple[T, list[T]]) -> tuple[T, list[T]]:
    """Plotkin reduction."""
    head, body = c
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            smaller = body[:i] + body[i + 1 :]
            if subsumes((head, body), (head, smaller)):
                body = smaller
                changed = True
                break
    return head, body


def rlgg(examples: list[str], background: list[str]) -> str:
    """The relative lgg, head-connected and reduced, as clause text."""
    bk = [parse(b) for b in background]
    acc: tuple[T, list[T]] = (parse(examples[0]), bk)
    for e in examples[1:]:
        acc = lgg_clauses(acc, (parse(e), bk))
    head, body = acc
    head, body = reduce((head, connected(head, body)))
    return show(head) + (" :- " + ", ".join(show(b) for b in body) if body else "") + "."


# ------------------------------------------------------------------------------------------------ version spaces

type H = tuple[str | None, ...]


def covers(h: H, x: Sequence[str]) -> bool:
    """Whether a conjunctive hypothesis covers an instance."""
    return all(c == "?" or (c is not None and c == v) for c, v in zip(h, x, strict=True))


def more_general(a: H, b: H) -> bool:
    """a >= b over instances."""
    if any(c is None for c in b):
        return True
    return all(x == "?" or (x is not None and x == y) for x, y in zip(a, b, strict=True))


def brute_version_space(
    examples: list[tuple[list[str], bool]], domains: list[list[str]]
) -> tuple[list[list[str]], list[list[str]], int]:
    """S and G by enumerating every consistent (non-empty) hypothesis; also the version space's size."""
    consistent: list[H] = []
    empty: H = tuple(None for _ in domains)
    for h in [empty, *product(*[["?", *d] for d in domains])]:
        if all(covers(h, x) == y for x, y in examples):
            consistent.append(h)
    s = [h for h in consistent if not any(o != h and more_general(h, o) for o in consistent)]
    g = [h for h in consistent if not any(o != h and more_general(o, h) for o in consistent)]

    def as_list(h: H) -> list[str]:
        return [c if c is not None else "∅" for c in h]

    return sorted(as_list(h) for h in s), sorted(as_list(h) for h in g), len(consistent)


def find_s(examples: list[tuple[list[str], bool]]) -> list[str | None]:
    """Mitchell's Find-S."""
    n = len(examples[0][0])
    h: list[str | None] = [None] * n
    for x, y in examples:
        if not y:
            continue
        for i, v in enumerate(x):
            h[i] = v if h[i] is None else (h[i] if h[i] == v else "?")
    return h


# ------------------------------------------------------------------------------------------------ FOIL


@dataclass(frozen=True)
class Lit:
    """A body literal over variable indices."""

    pred: str
    args: tuple[int, ...]


def gain(p0: int, n0: int, p1: int, n1: int, t: int) -> float:
    """Quinlan's information gain."""
    if p1 == 0 or p0 == 0:
        return 0.0
    return t * (math.log2(p1 / (p1 + n1)) - math.log2(p0 / (p0 + n0)))


def foil(
    target: str,
    arity: int,
    positive: list[tuple[str, ...]],
    background: dict[str, list[tuple[str, ...]]],
    negative: list[tuple[str, ...]] | None = None,
    recursion: bool = False,
    max_body: int = 4,
    max_new: int = 1,
) -> tuple[list[str], list[float]]:
    """Learned clauses (text, variables A, B, ...) and the gain of each literal chosen."""
    consts = sorted(
        {c for t in positive for c in t}
        | {c for t in negative or [] for c in t}
        | {c for rel in background.values() for t in rel for c in t}
    )
    pos_set = set(positive)
    if negative is None:
        negative = [t for t in product(consts, repeat=arity) if t not in pos_set]
    rels = dict(sorted(background.items()))
    preds = [(p, len(ts[0]) if ts else 0) for p, ts in rels.items()]
    if recursion:
        rels[target] = list(positive)
        preds.append((target, arity))

    def extend(lit: Lit, tup: tuple[str, ...]) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        for m in rels[lit.pred]:
            nxt = list(tup)
            ok = True
            for a, v in zip(lit.args, m, strict=True):
                if a < len(tup):
                    if tup[a] != v:
                        ok = False
                elif a < len(nxt):
                    if nxt[a] != v:
                        ok = False
                else:
                    nxt.append(v)
            if ok:
                out.append(tuple(nxt))
        return out

    def candidates(nvars: int, body: list[Lit]) -> list[Lit]:
        out: list[Lit] = []
        for p, m in preds:

            def rec(prefix: list[int], nxt: int, p: str = p, m: int = m) -> None:
                if len(prefix) == m:
                    if not any(a < nvars for a in prefix):
                        return
                    # A recursive literal uses bound variables only and differs from the head.
                    if p == target and (any(a >= nvars for a in prefix) or all(a == i for i, a in enumerate(prefix))):
                        return
                    lit = Lit(p, tuple(prefix))
                    if lit not in body:
                        out.append(lit)
                    return
                for a in range(nvars):
                    rec([*prefix, a], nxt)
                if nxt - nvars < max_new:
                    rec([*prefix, nxt], nxt + 1)

            rec([], nvars)
        return out

    remaining = list(positive)
    clauses: list[str] = []
    gains: list[float] = []
    while remaining:
        body: list[Lit] = []
        nvars = arity
        pos = list(remaining)
        neg = list(negative)
        while neg and len(body) < max_body:
            best: tuple[float, int, int, Lit, list[tuple[str, ...]], list[tuple[str, ...]]] | None = None
            for order, lit in enumerate(candidates(nvars, body)):
                t = 0
                p1: list[tuple[str, ...]] = []
                for tup in pos:
                    ext = extend(lit, tup)
                    if ext:
                        t += 1
                    p1.extend(ext)
                n1 = [e for tup in neg for e in extend(lit, tup)]
                fresh = sum(1 for a in lit.args if a >= nvars)
                g = gain(len(pos), len(neg), len(p1), len(n1), t)
                key = (g, -fresh, -order)
                if best is None or key > (best[0], -best[1], -best[2]):
                    best = (g, fresh, order, lit, p1, n1)
            if best is None or best[0] <= 1e-12:
                return clauses, gains
            gains.append(best[0])
            body.append(best[3])
            nvars += best[1]
            pos, neg = best[4], best[5]
        if neg:
            return clauses, gains
        covered = {t[:arity] for t in pos}
        remaining = [t for t in remaining if t not in covered]

        def name(i: int) -> str:
            return chr(65 + i)

        head = f"{target}({','.join(name(i) for i in range(arity))})"
        lits = [f"{lit.pred}({','.join(name(a) for a in lit.args)})" for lit in body]
        clauses.append(head + (" :- " + ", ".join(lits) if lits else "") + ".")
    return clauses, gains
