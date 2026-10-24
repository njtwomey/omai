"""A reference bottom-up Datalog evaluator (naive iteration per stratum), for round counts: clingo gives models but not
the rounds of the immediate consequence operator. Rules are read with clingo's own parser-independent syntax of the
fixtures (Prolog-style terms), stratified by the least levels (a predicate sits above every predicate it negates)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product

TOKEN = re.compile(r"\s*(\\\+|not\b|:-|\\=|=<|>=|==|[(),.<>=]|[A-Za-z_][A-Za-z0-9_]*|-?\d+)")


@dataclass(frozen=True)
class Lit:
    """A body literal: kind is pos, neg or cmp; for cmp, pred is the operator and args the two sides."""

    kind: str
    pred: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    """A rule over string terms (variables start upper case or with an underscore)."""

    pred: str
    args: tuple[str, ...]
    body: tuple[Lit, ...]


def tokens(src: str) -> list[str]:
    """Tokens, comments removed."""
    src = re.sub(r"%[^\n]*", "", src)
    out: list[str] = []
    pos = 0
    while pos < len(src):
        if src[pos].isspace():
            pos += 1
            continue
        m = TOKEN.match(src, pos)
        if m is None:
            raise SyntaxError(src[pos : pos + 20])
        out.append(m.group(1))
        pos = m.end()
    return out


def parse(src: str) -> list[Rule]:
    """Rules of a program."""
    toks = tokens(src)
    i = 0

    def atom() -> tuple[str, tuple[str, ...]]:
        nonlocal i
        name = toks[i]
        i += 1
        args: list[str] = []
        if i < len(toks) and toks[i] == "(":
            i += 1
            while True:
                args.append(toks[i])
                i += 1
                if toks[i] == ",":
                    i += 1
                    continue
                i += 1
                break
        return name, tuple(args)

    rules: list[Rule] = []
    while i < len(toks):
        pred, args = atom()
        body: list[Lit] = []
        if toks[i] == ":-":
            i += 1
            while True:
                if toks[i] in ("not", "\\+"):
                    i += 1
                    p, a = atom()
                    body.append(Lit("neg", p, a))
                elif i + 1 < len(toks) and toks[i + 1] in ("<", ">", "=<", ">=", "=", "\\=", "=="):
                    left, op, right = toks[i], toks[i + 1], toks[i + 2]
                    i += 3
                    body.append(Lit("cmp", op, (left, right)))
                else:
                    p, a = atom()
                    body.append(Lit("pos", p, a))
                if toks[i] == ",":
                    i += 1
                    continue
                break
        assert toks[i] == ".", toks[i : i + 3]
        i += 1
        rules.append(Rule(pred, args, tuple(body)))
    return rules


def key(pred: str, n: int) -> str:
    """A predicate indicator."""
    return f"{pred}/{n}"


def strata(rules: list[Rule]) -> list[list[str]]:
    """Least stratification; raises on recursion through negation."""
    preds = {key(r.pred, len(r.args)) for r in rules} | {
        key(lit.pred, len(lit.args)) for r in rules for lit in r.body if lit.kind != "cmp"
    }
    level = dict.fromkeys(preds, 0)
    for _ in range(len(preds) + 2):
        changed = False
        for r in rules:
            h = key(r.pred, len(r.args))
            for lit in r.body:
                if lit.kind == "cmp":
                    continue
                need = level[key(lit.pred, len(lit.args))] + (1 if lit.kind == "neg" else 0)
                if need > level[h]:
                    level[h] = need
                    changed = True
        if not changed:
            top = max(level.values(), default=0)
            return [
                sorted(p for p in preds if level[p] == k) for k in range(top + 1) if any(level[p] == k for p in preds)
            ]
    raise ValueError("not stratifiable")


def is_var(t: str) -> bool:
    """Whether a term is a variable."""
    return t[0].isupper() or t[0] == "_"


def value(t: str) -> int | str:
    """A constant's comparable value."""
    return int(t) if re.fullmatch(r"-?\d+", t) else t


def holds(op: str, a: str, b: str) -> bool:
    """A comparison of constants."""
    x, y = value(a), value(b)
    if op in ("=", "=="):
        return x == y
    if op == "\\=":
        return x != y
    assert isinstance(x, int) and isinstance(y, int)
    return {"<": x < y, ">": x > y, "=<": x <= y, ">=": x >= y}[op]


Fact = tuple[str, tuple[str, ...]]


def consequences(rule: Rule, facts: set[Fact]) -> set[Fact]:
    """Every head instance whose body holds in `facts` (negation and comparisons checked once ground)."""
    out: set[Fact] = set()
    positives = [lit for lit in rule.body if lit.kind == "pos"]
    pools = [[f for f in facts if f[0] == lit.pred and len(f[1]) == len(lit.args)] for lit in positives]
    for combo in product(*pools):
        b: dict[str, str] = {}
        ok = True
        for lit, f in zip(positives, combo, strict=True):
            for t, v in zip(lit.args, f[1], strict=True):
                if is_var(t):
                    if t == "_" or t.startswith("_"):
                        continue
                    if b.setdefault(t, v) != v:
                        ok = False
                elif t != v:
                    ok = False
        if not ok:
            continue

        def sub(t: str, b: dict[str, str] = b) -> str:
            return b.get(t, t) if is_var(t) else t

        for lit in rule.body:
            if lit.kind == "neg" and (lit.pred, tuple(sub(a) for a in lit.args)) in facts:
                ok = False
            if lit.kind == "cmp" and not holds(lit.pred, sub(lit.args[0]), sub(lit.args[1])):
                ok = False
        if ok:
            out.add((rule.pred, tuple(sub(a) for a in rule.args)))
    return out


def evaluate(src: str) -> tuple[list[str], list[list[int]]]:
    """The model (sorted fact texts) and, per stratum, the new facts of each round (ending with an empty round)."""
    rules = parse(src)
    facts: set[Fact] = {(r.pred, r.args) for r in rules if not r.body}
    rounds: list[list[int]] = []
    for stratum in strata(rules):
        mine = [r for r in rules if r.body and key(r.pred, len(r.args)) in stratum]
        counts: list[int] = []
        while True:
            new: set[Fact] = set()
            for r in mine:
                new |= consequences(r, facts)
            new -= facts
            counts.append(len(new))
            facts |= new
            if not new:
                break
        rounds.append(counts)
    text = sorted(f"{p}({','.join(a)})" if a else p for p, a in facts)
    return text, rounds
