"""A small reference Prolog interpreter (SWI-Prolog is not packaged for uv): a typed port of the classic
generator-based design, independent of the engine's continuation machine.

Solutions are Python generators over a global binding trail; cut is a flag on the clause activation that every
conjunction and disjunction checks when its inner generator ends, so `!` behaves as in ISO Prolog (if-then-else
conditions, negation, call/N and findall are opaque to it). The built-ins and the library predicates cover what the
fixtures use, with SWI-Prolog's defaults (integer arithmetic stays integral, `/` of integers is exact when it can be,
double-quoted text is a list of codes). The standard library is SWI's own Prolog definitions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------ terms


@dataclass(frozen=True)
class Atom:
    """A constant."""

    name: str


@dataclass(frozen=True)
class Int:
    """An integer."""

    value: int


@dataclass(frozen=True)
class Float:
    """A float."""

    value: float


class Var:
    """A variable; `ref` holds its binding."""

    __slots__ = ("name", "ref")

    def __init__(self, name: str) -> None:
        self.name = name
        self.ref: Term | None = None


@dataclass(frozen=True)
class Struct:
    """A compound term."""

    name: str
    args: tuple[Term, ...]


type Term = Atom | Int | Float | Var | Struct

NIL = Atom("[]")
NO_ARGS: tuple[Term, ...] = ()
TRUE = Atom("true")


def mklist(items: list[Term], tail: Term = NIL) -> Term:
    """A list term."""
    out = tail
    for x in reversed(items):
        out = Struct(".", (x, out))
    return out


def deref(t: Term) -> Term:
    """Follow bindings."""
    while isinstance(t, Var) and t.ref is not None:
        t = t.ref
    return t


def resolve(t: Term) -> Term:
    """Apply bindings fully."""
    t = deref(t)
    if isinstance(t, Struct):
        return Struct(t.name, tuple(resolve(a) for a in t.args))
    return t


def list_items(t: Term) -> list[Term] | None:
    """The elements of a proper list, or None."""
    out: list[Term] = []
    t = deref(t)
    while isinstance(t, Struct) and t.name == "." and len(t.args) == 2:
        out.append(t.args[0])
        t = deref(t.args[1])
    return out if t == NIL else None


# ------------------------------------------------------------------------------------------------ errors


class PrologError(Exception):
    """A thrown ball."""

    def __init__(self, ball: Term) -> None:
        super().__init__("prolog error")
        self.ball = ball


class Budget(Exception):
    """The inference budget ran out."""


def err(formal: Term) -> PrologError:
    """An ISO error term."""
    return PrologError(Struct("error", (formal, Var("_"))))


def type_error(kind: str, culprit: Term) -> PrologError:
    """type_error(Kind, Culprit)."""
    return err(Struct("type_error", (Atom(kind), resolve(culprit))))


def inst_error() -> PrologError:
    """instantiation_error."""
    return err(Atom("instantiation_error"))


# ------------------------------------------------------------------------------------------------ reader

SYMBOL = "+-*/\\^<>=~:.?@#&$"
INFIX: dict[str, tuple[int, str]] = {}
PREFIX: dict[str, tuple[int, str]] = {}
for pri, typ, names in [
    (1200, "xfx", ":- -->"),
    (1100, "xfy", "; |"),
    (1050, "xfy", "-> *->"),
    (1000, "xfy", ","),
    (700, "xfx", "= \\= == \\== @< @> @=< @>= =.. is =:= =\\= < > =< >="),
    (600, "xfy", ":"),
    (500, "yfx", "+ - /\\ \\/ xor"),
    (400, "yfx", "* / // mod rem div << >>"),
    (200, "xfx", "**"),
    (200, "xfy", "^"),
]:
    for n in names.split():
        INFIX[n] = (pri, typ)
for pri, typ, names in [(1200, "fx", ":- ?-"), (1150, "fx", "dynamic"), (900, "fy", "\\+"), (200, "fy", "- + \\")]:
    for n in names.split():
        PREFIX[n] = (pri, typ)


@dataclass
class Tok:
    """A token: kind is atom, var, int, float, str, punct or end."""

    kind: str
    text: str
    layout: bool


ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"', "0": "\0"}


def tokenize(src: str) -> list[Tok]:
    """Split Prolog text into tokens."""
    out: list[Tok] = []
    i = 0
    layout = True
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            layout = True
            i += 1
            continue
        if c == "%":
            while i < n and src[i] != "\n":
                i += 1
            layout = True
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i + 2) + 2
            layout = True
            continue
        start = i
        if c.isdigit():
            if src.startswith("0'", i):
                ch = src[i + 2]
                if ch == "\\":
                    out.append(Tok("int", str(ord(ESCAPES[src[i + 3]])), layout))
                    i += 4
                else:
                    out.append(Tok("int", str(ord(ch)), layout))
                    i += 3
                layout = False
                continue
            while i < n and src[i].isdigit():
                i += 1
            kind = "int"
            if i + 1 < n and src[i] == "." and src[i + 1].isdigit():
                kind = "float"
                i += 1
                while i < n and src[i].isdigit():
                    i += 1
            if i < n and src[i] in "eE" and i + 1 < n and (src[i + 1].isdigit() or src[i + 1] in "+-"):
                kind = "float"
                i += 2
                while i < n and src[i].isdigit():
                    i += 1
            out.append(Tok(kind, src[start:i], layout))
        elif c == "_" or c.isupper():
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            out.append(Tok("var", src[start:i], layout))
        elif c.islower():
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            out.append(Tok("atom", src[start:i], layout))
        elif c in "'\"":
            i += 1
            s = ""
            while True:
                d = src[i]
                if d == c:
                    if i + 1 < n and src[i + 1] == c:
                        s += c
                        i += 2
                        continue
                    i += 1
                    break
                if d == "\\":
                    s += ESCAPES[src[i + 1]]
                    i += 2
                    continue
                s += d
                i += 1
            out.append(Tok("atom" if c == "'" else "str", s, layout))
        elif c in "()[]{},|":
            out.append(Tok("punct", c, layout))
            i += 1
        elif c in "!;":
            out.append(Tok("atom", c, layout))
            i += 1
        elif c in SYMBOL:
            if c == "." and (i + 1 >= n or src[i + 1].isspace() or src[i + 1] == "%"):
                out.append(Tok("end", ".", layout))
                i += 1
            else:
                while i < n and src[i] in SYMBOL:
                    i += 1
                out.append(Tok("atom", src[start:i], layout))
        else:
            raise SyntaxError(f"unexpected {c!r}")
        layout = False
    return out


class Reader:
    """Operator precedence parser over tokens; variables are shared by name within a clause."""

    def __init__(self, toks: list[Tok]) -> None:
        self.toks = toks
        self.pos = 0
        self.vars: dict[str, Var] = {}
        self.names: list[str] = []

    def peek(self, k: int = 0) -> Tok | None:
        """The token k ahead."""
        j = self.pos + k
        return self.toks[j] if j < len(self.toks) else None

    def take(self) -> Tok:
        """Consume a token."""
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def expect(self, text: str) -> None:
        """Consume a given punctuation token."""
        t = self.take()
        if t.text != text:
            raise SyntaxError(f"expected {text}, found {t.text}")

    def clause(self) -> Term:
        """Read up to an end token."""
        self.vars = {}
        self.names = []
        t = self.parse(1200)
        if self.take().kind != "end":
            raise SyntaxError("operator expected")
        return t

    def parse(self, mx: int) -> Term:
        """Read a term of priority at most mx."""
        left, lp = self.primary(mx)
        while True:
            t = self.peek()
            if t is None or not (t.kind == "atom" or (t.kind == "punct" and t.text in ",|")):
                break
            op = INFIX.get(t.text)
            if op is None:
                break
            p, typ = op
            la = p if typ == "yfx" else p - 1
            ra = p if typ == "xfy" else p - 1
            if p > mx or lp > la:
                break
            self.pos += 1
            right = self.parse(ra)
            left = Struct(";" if t.text == "|" else t.text, (left, right))
            lp = p
        return left

    def arglist(self) -> list[Term]:
        """Comma-separated arguments at priority 999."""
        args = [self.parse(999)]
        while (t := self.peek()) is not None and t.kind == "punct" and t.text == ",":
            self.pos += 1
            args.append(self.parse(999))
        return args

    def var(self, name: str) -> Var:
        """The clause's variable of that name (`_` is always fresh)."""
        if name == "_":
            return Var("_")
        v = self.vars.get(name)
        if v is None:
            v = Var(name)
            self.vars[name] = v
            self.names.append(name)
        return v

    def primary(self, mx: int) -> tuple[Term, int]:
        """A primary term and its priority."""
        t = self.take()
        if t.kind == "int":
            return Int(int(t.text)), 0
        if t.kind == "float":
            return Float(float(t.text)), 0
        if t.kind == "var":
            return self.var(t.text), 0
        if t.kind == "str":
            return mklist([Int(ord(ch)) for ch in t.text]), 0
        if t.kind == "punct":
            if t.text == "(":
                inner = self.parse(1200)
                self.expect(")")
                return inner, 0
            if t.text == "[":
                nt = self.peek()
                if nt is not None and nt.text == "]":
                    self.pos += 1
                    return NIL, 0
                items = self.arglist()
                tail: Term = NIL
                nt = self.peek()
                if nt is not None and nt.text == "|":
                    self.pos += 1
                    tail = self.parse(999)
                self.expect("]")
                return mklist(items, tail), 0
            if t.text == "{":
                inner = self.parse(1200)
                self.expect("}")
                return Struct("{}", (inner,)), 0
            raise SyntaxError(f"unexpected {t.text}")
        name = t.text
        nt = self.peek()
        if nt is not None and nt.kind == "punct" and nt.text == "(" and not nt.layout:
            self.pos += 1
            args = self.arglist()
            self.expect(")")
            return Struct(name, tuple(args)), 0
        if name == "-" and nt is not None and nt.kind in ("int", "float") and not nt.layout:
            self.pos += 1
            return (Int(-int(nt.text)) if nt.kind == "int" else Float(-float(nt.text))), 0
        pre = PREFIX.get(name)
        starts = nt is not None and nt.kind != "end" and not (nt.kind == "punct" and nt.text in ")]},|")
        if starts and nt is not None and nt.kind == "atom" and nt.text in INFIX and nt.text not in PREFIX:
            n2 = self.peek(1)
            starts = n2 is not None and n2.kind == "punct" and n2.text == "(" and not n2.layout
        if pre is not None and starts:
            p, typ = pre
            arg_max = p if typ == "fy" else p - 1
            if p > mx:
                p, arg_max = 999, 999
            return Struct(name, (self.parse(arg_max),)), p
        return Atom(name), 0


def read_program(src: str) -> list[Term]:
    """Every clause term of a program."""
    r = Reader(tokenize(src))
    out: list[Term] = []
    while r.peek() is not None:
        out.append(r.clause())
    return out


def read_query(src: str) -> tuple[Term, list[str], dict[str, Var]]:
    """A query term with its variable names in order of first occurrence."""
    r = Reader(tokenize(src.strip().rstrip(".") + " ."))
    t = r.clause()
    return t, r.names, r.vars


# ------------------------------------------------------------------------------------------------ writer


def plain(name: str) -> bool:
    """Whether an atom needs no quotes."""
    if name in ("[]", "!", ";", "{}"):
        return True
    if not name:
        return False
    if name[0].islower():
        return all(ch.isalnum() or ch == "_" for ch in name)
    return all(ch in SYMBOL for ch in name)


def quote(name: str) -> str:
    """An atom as writeq writes it."""
    if plain(name):
        return name
    body = name.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\t", "\\t")
    return f"'{body}'"


def fmt_float(x: float) -> str:
    """A float as the engine writes it."""
    s = repr(x)
    return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"


def canonical(t: Term, names: dict[int, str]) -> str:
    """Canonical text (no operators, lists in brackets); variables named by `names` (by identity), else `_`."""
    t = deref(t)
    if isinstance(t, Var):
        return names.get(id(t), "_")
    if isinstance(t, Atom):
        return quote(t.name)
    if isinstance(t, Int):
        return str(t.value)
    if isinstance(t, Float):
        return fmt_float(t.value)
    if t.name == "." and len(t.args) == 2:
        parts: list[str] = []
        cur: Term = t
        while isinstance(cur, Struct) and cur.name == "." and len(cur.args) == 2:
            parts.append(canonical(cur.args[0], names))
            cur = deref(cur.args[1])
        tail = "" if cur == NIL else "|" + canonical(cur, names)
        return "[" + ",".join(parts) + tail + "]"
    return quote(t.name) + "(" + ",".join(canonical(a, names) for a in t.args) + ")"


def write_text(t: Term, quoted: bool) -> str:
    """write/1 (and print/1 when quoted) for the subset the fixtures print: canonical form with `_` variables."""
    t = deref(t)
    if isinstance(t, Atom) and not quoted:
        return t.name
    return canonical(t, {}) if quoted else canonical_unquoted(t)


def canonical_unquoted(t: Term) -> str:
    """Canonical text without quotes."""
    t = deref(t)
    if isinstance(t, Atom):
        return t.name
    if isinstance(t, Struct):
        if t.name == "." and len(t.args) == 2:
            items = list_items(t)
            if items is not None:
                return "[" + ",".join(canonical_unquoted(a) for a in items) + "]"
        return t.name + "(" + ",".join(canonical_unquoted(a) for a in t.args) + ")"
    return canonical(t, {})


# ------------------------------------------------------------------------------------------------ the solver

LIBRARY = """
append([], L, L).
append([H|T], L, [H|R]) :- append(T, L, R).
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).
memberchk(X, L) :- member(X, L), !.
reverse(Xs, Ys) :- '$reverse'(Xs, Ys, [], Ys).
'$reverse'([], [], Ys, Ys).
'$reverse'([X|Xs], [_|Bound], Rs, Ys) :- '$reverse'(Xs, Bound, [X|Rs], Ys).
nth0(I, L, E) :- integer(I), !, I >= 0, '$nth'(I, L, E).
nth0(I, L, E) :- var(I), !, '$nth_gen'(L, E, 0, I).
nth1(I, L, E) :- integer(I), !, I >= 1, I0 is I - 1, '$nth'(I0, L, E).
nth1(I, L, E) :- var(I), !, '$nth_gen'(L, E, 1, I).
'$nth'(I, [H|T], E) :- ( I =:= 0 -> E = H ; I1 is I - 1, '$nth'(I1, T, E) ).
'$nth_gen'([E|_], E, B, B).
'$nth_gen'([_|T], E, B0, B) :- B1 is B0 + 1, '$nth_gen'(T, E, B1, B).
last([X|Xs], Last) :- '$last'(Xs, X, Last).
'$last'([], Last, Last).
'$last'([X|Xs], _, Last) :- '$last'(Xs, X, Last).
select(X, [X|T], T).
select(X, [H|T], [H|R]) :- select(X, T, R).
subtract([], _, []).
subtract([E|T], D, R) :- memberchk(E, D), !, subtract(T, D, R).
subtract([H|T], D, [H|R]) :- subtract(T, D, R).
intersection([], _, []).
intersection([X|T], L, I) :- memberchk(X, L), !, I = [X|R], intersection(T, L, R).
intersection([_|T], L, R) :- intersection(T, L, R).
union([], L, L).
union([H|T], L, R) :- memberchk(H, L), !, union(T, L, R).
union([H|T], L, [H|R]) :- union(T, L, R).
delete([], _, []).
delete([E|T], D, R) :- E \\= D, !, R = [E|R1], delete(T, D, R1).
delete([_|T], D, R) :- delete(T, D, R).
exclude(_, [], []).
exclude(P, [H|T], R) :- ( call(P, H) -> R = R1 ; R = [H|R1] ), exclude(P, T, R1).
include(_, [], []).
include(P, [H|T], R) :- ( call(P, H) -> R = [H|R1] ; R = R1 ), include(P, T, R1).
maplist(_, []).
maplist(P, [A|As]) :- call(P, A), maplist(P, As).
maplist(_, [], []).
maplist(P, [A|As], [B|Bs]) :- call(P, A, B), maplist(P, As, Bs).
foldl(G, L, V0, V) :- '$foldl'(L, G, V0, V).
'$foldl'([], _, V, V).
'$foldl'([X|Xs], G, V0, V) :- call(G, X, V0, V1), '$foldl'(Xs, G, V1, V).
sum_list(Xs, Sum) :- '$sum_list'(Xs, 0, Sum).
'$sum_list'([], Sum, Sum).
'$sum_list'([X|Xs], Sum0, Sum) :- Sum1 is Sum0 + X, '$sum_list'(Xs, Sum1, Sum).
max_list([H|T], Max) :- '$max_list'(T, H, Max).
'$max_list'([], Max, Max).
'$max_list'([H|T], Max0, Max) :- Max1 is max(H, Max0), '$max_list'(T, Max1, Max).
min_list([H|T], Min) :- '$min_list'(T, H, Min).
'$min_list'([], Min, Min).
'$min_list'([H|T], Min0, Min) :- Min1 is min(H, Min0), '$min_list'(T, Min1, Min).
numlist(L, H, R) :- L =< H, '$numlist'(L, H, R).
'$numlist'(H, H, [H]) :- !.
'$numlist'(L, H, [L|T]) :- L1 is L + 1, '$numlist'(L1, H, T).
permutation([], []).
permutation(L, [X|P]) :- select(X, L, R), permutation(R, P).
once(G) :- call(G), !.
ignore(G) :- ( call(G) -> true ; true ).
forall(C, A) :- \\+ ( call(C), \\+ call(A) ).
aggregate_all(count, G, C) :- !, findall(x, G, L), length(L, C).
aggregate_all(sum(E), G, S) :- !, findall(E, G, L), sum_list(L, S).
aggregate_all(max(E), G, M) :- !, findall(E, G, L), L = [_|_], max_list(L, M).
aggregate_all(bag(E), G, L) :- !, findall(E, G, L).
aggregate_all(set(E), G, S) :- !, findall(E, G, L), sort(L, S).
"""


def kind_rank(t: Term) -> int:
    """Standard order class: Var < Number < Atom < Compound."""
    if isinstance(t, Var):
        return 0
    if isinstance(t, Int | Float):
        return 1
    if isinstance(t, Atom):
        return 2
    return 3


def compare(a: Term, b: Term) -> int:
    """The standard order of terms."""
    a, b = deref(a), deref(b)
    ka, kb = kind_rank(a), kind_rank(b)
    if ka != kb:
        return -1 if ka < kb else 1
    if isinstance(a, Var) and isinstance(b, Var):
        return 0 if a is b else (-1 if id(a) < id(b) else 1)
    if isinstance(a, Int | Float) and isinstance(b, Int | Float):
        if a.value != b.value:
            return -1 if a.value < b.value else 1
        fa, fb = isinstance(a, Float), isinstance(b, Float)
        return 0 if fa == fb else (-1 if fa else 1)
    if isinstance(a, Atom) and isinstance(b, Atom):
        return 0 if a.name == b.name else (-1 if a.name < b.name else 1)
    assert isinstance(a, Struct) and isinstance(b, Struct)
    if len(a.args) != len(b.args):
        return -1 if len(a.args) < len(b.args) else 1
    if a.name != b.name:
        return -1 if a.name < b.name else 1
    for x, y in zip(a.args, b.args, strict=True):
        c = compare(x, y)
        if c:
            return c
    return 0


def sort_terms(items: list[Term], dedupe: bool) -> list[Term]:
    """Sort by the standard order (stable), optionally removing duplicates."""
    out: list[Term] = []
    for x in items:
        out.append(x)
    # insertion-free merge sort via functools would need a key; a simple stable sort with cmp_to_key.
    from functools import cmp_to_key

    out.sort(key=cmp_to_key(compare))
    if dedupe:
        uniq: list[Term] = []
        for x in out:
            if not uniq or compare(uniq[-1], x) != 0:
                uniq.append(x)
        return uniq
    return out


@dataclass
class Frame:
    """A clause activation (or an opaque call): cut sets the flag."""

    cut: bool = False


def num_value(t: Term) -> Int | Float:
    """A number term or a type error."""
    if isinstance(t, Int | Float):
        return t
    raise type_error("evaluable", t)


def evaluate(t: Term) -> Int | Float:
    """Arithmetic with SWI-Prolog's defaults."""
    t = deref(t)
    if isinstance(t, Var):
        raise inst_error()
    if isinstance(t, Int | Float):
        return t
    if isinstance(t, Atom):
        if t.name == "pi":
            import math

            return Float(math.pi)
        raise type_error("evaluable", Struct("/", (t, Int(0))))
    vals = [evaluate(a) for a in t.args]
    ints = all(isinstance(v, Int) for v in vals)
    key = (t.name, len(vals))
    if len(vals) == 2:
        x, y = vals[0].value, vals[1].value
        if key == ("+", 2):
            return Int(int(x + y)) if ints else Float(float(x + y))
        if key == ("-", 2):
            return Int(int(x - y)) if ints else Float(float(x - y))
        if key == ("*", 2):
            return Int(int(x * y)) if ints else Float(float(x * y))
        if key == ("/", 2):
            if y == 0:
                raise err(Struct("evaluation_error", (Atom("zero_divisor"),)))
            if ints and int(x) % int(y) == 0:
                return Int(int(x) // int(y))
            return Float(x / y)
        if key in (("//", 2), ("mod", 2), ("rem", 2), ("div", 2)):
            if not ints:
                raise type_error("integer", t.args[0] if isinstance(vals[0], Float) else t.args[1])
            a, b = int(x), int(y)
            if b == 0:
                raise err(Struct("evaluation_error", (Atom("zero_divisor"),)))
            q = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
            if t.name == "//":
                return Int(q)
            if t.name == "rem":
                return Int(a - b * q)
            if t.name == "div":
                return Int(a // b)
            return Int(a % b)
        if key == ("min", 2):
            return vals[1] if y < x else vals[0]
        if key == ("max", 2):
            return vals[1] if y > x else vals[0]
        if key == ("**", 2):
            if ints and y >= 0:
                return Int(int(x) ** int(y))
            return Float(float(x) ** float(y))
        if key == ("^", 2):
            if ints:
                if y < 0:
                    raise type_error("float", t.args[0])
                return Int(int(x) ** int(y))
            return Float(float(x) ** float(y))
    if len(vals) == 1:
        x = vals[0].value
        if key == ("-", 1):
            return Int(-int(x)) if ints else Float(-float(x))
        if key == ("abs", 1):
            return Int(abs(int(x))) if ints else Float(abs(float(x)))
        if key == ("float", 1):
            return Float(float(x))
        if key == ("truncate", 1):
            return Int(int(x))
    raise type_error("evaluable", Struct("/", (Atom(t.name), Int(len(vals)))))


class Machine:
    """A database and the solver over it."""

    def __init__(self, program: str, max_inferences: int = 2_000_000, max_depth: int = 10_000) -> None:
        self.db: dict[tuple[str, int], list[Term]] = {}
        self.dynamic: set[tuple[str, int]] = set()
        self.trail: list[Var] = []
        self.output = ""
        self.inferences = 0
        self.max_inferences = max_inferences
        self.max_depth = max_depth
        self.depth_limited = False
        self.counter = 0
        user: set[tuple[str, int]] = set()
        for t in read_program(LIBRARY):
            self.add(t, False)
        for t in read_program(program):
            if isinstance(t, Struct) and t.name == ":-" and len(t.args) == 1:
                d = deref(t.args[0])
                if isinstance(d, Struct) and d.name == "dynamic":
                    spec = deref(d.args[0])
                    if isinstance(spec, Struct) and spec.name == "/":
                        n, a = deref(spec.args[0]), deref(spec.args[1])
                        if isinstance(n, Atom) and isinstance(a, Int):
                            self.dynamic.add((n.name, a.value))
                            self.db.setdefault((n.name, a.value), [])
                continue
            k = self.key(head_of(t))
            if k not in user:
                user.add(k)
                self.db[k] = []
            self.add(t, False)

    @staticmethod
    def key(t: Term) -> tuple[str, int]:
        """The predicate indicator."""
        t = deref(t)
        if isinstance(t, Atom):
            return (t.name, 0)
        if isinstance(t, Struct):
            return (t.name, len(t.args))
        if isinstance(t, Var):
            raise inst_error()
        raise type_error("callable", t)

    def add(self, t: Term, front: bool) -> None:
        """Store a clause term."""
        k = self.key(head_of(t))
        clauses = self.db.get(k, [])
        self.db[k] = [t, *clauses] if front else [*clauses, t]

    # bindings

    def bind(self, v: Var, t: Term) -> None:
        """Bind and trail."""
        v.ref = t
        self.trail.append(v)

    def undo(self, mark: int) -> None:
        """Unwind the trail."""
        while len(self.trail) > mark:
            self.trail.pop().ref = None

    def unify(self, a: Term, b: Term) -> bool:
        """Unification (no occurs check); a variable on the left binds to the right."""
        stack: list[tuple[Term, Term]] = [(a, b)]
        while stack:
            x, y = stack.pop()
            x, y = deref(x), deref(y)
            if x is y:
                continue
            if isinstance(x, Var):
                self.bind(x, y)
            elif isinstance(y, Var):
                self.bind(y, x)
            elif isinstance(x, Struct) and isinstance(y, Struct):
                if x.name != y.name or len(x.args) != len(y.args):
                    return False
                stack.extend(reversed(list(zip(x.args, y.args, strict=True))))
            elif x != y:
                return False
        return True

    def fresh(self) -> Var:
        """A new variable."""
        self.counter += 1
        return Var(f"_{self.counter}")

    def copy(self, t: Term, names: dict[int, Var] | None = None) -> Term:
        """Copy with fresh variables."""
        names = {} if names is None else names
        t = deref(t)
        if isinstance(t, Var):
            v = names.get(id(t))
            if v is None:
                v = self.fresh()
                names[id(t)] = v
            return v
        if isinstance(t, Struct):
            return Struct(t.name, tuple(self.copy(a, names) for a in t.args))
        return t

    # solving

    def solve(self, goal: Term, frame: Frame, depth: int) -> Iterator[None]:
        """Prove a goal; yields once per solution."""
        self.inferences += 1
        if self.inferences > self.max_inferences:
            raise Budget()
        g = deref(goal)
        if isinstance(g, Var):
            raise inst_error()
        if isinstance(g, Int | Float):
            raise type_error("callable", g)
        name = g.name
        args = g.args if isinstance(g, Struct) else NO_ARGS
        k = (name, len(args))
        if k == ("true", 0):
            yield
        elif k in (("fail", 0), ("false", 0)):
            return
        elif k == ("!", 0):
            yield
            frame.cut = True
        elif k == (",", 2):
            for _ in self.solve(args[0], frame, depth):
                yield from self.solve(args[1], frame, depth)
                if frame.cut:
                    return
        elif k == (";", 2):
            left = deref(args[0])
            if isinstance(left, Struct) and left.name == "->" and len(left.args) == 2:
                mark = len(self.trail)
                ok = False
                for _ in self.solve(left.args[0], Frame(), depth):
                    ok = True
                    break
                if ok:
                    yield from self.solve(left.args[1], frame, depth)
                else:
                    self.undo(mark)
                    yield from self.solve(args[1], frame, depth)
            else:
                yield from self.solve(args[0], frame, depth)
                if frame.cut:
                    return
                yield from self.solve(args[1], frame, depth)
        elif k == ("->", 2):
            ok = False
            for _ in self.solve(args[0], Frame(), depth):
                ok = True
                break
            if ok:
                yield from self.solve(args[1], frame, depth)
        elif k in (("\\+", 1), ("not", 1)):
            mark = len(self.trail)
            ok = False
            for _ in self.solve(args[0], Frame(), depth):
                ok = True
                break
            self.undo(mark)
            if not ok:
                yield
        elif name == "call" and k != ("call", 0):
            target = deref(args[0])
            extra = args[1:]
            if isinstance(target, Atom):
                target = Struct(target.name, extra) if extra else target
            elif isinstance(target, Struct):
                target = Struct(target.name, target.args + extra)
            elif isinstance(target, Var):
                raise inst_error()
            else:
                raise type_error("callable", target)
            yield from self.solve(target, Frame(), depth)
        elif k == ("findall", 3):
            results = self.collect(args[0], args[1], depth)
            mark = len(self.trail)
            if self.unify(args[2], mklist(results)):
                yield
            self.undo(mark)
        elif k in (("bagof", 3), ("setof", 3)):
            yield from self.bagof(args, name == "setof", depth)
        elif k == ("catch", 3):
            mark = len(self.trail)
            try:
                for _ in self.solve(args[0], Frame(), depth):
                    yield
            except PrologError as e:
                self.undo(mark)
                m2 = len(self.trail)
                if self.unify(args[1], e.ball):
                    yield from self.solve(args[2], Frame(), depth)
                    self.undo(m2)
                else:
                    self.undo(m2)
                    raise
        elif k == ("throw", 1):
            ball = resolve(args[0])
            if isinstance(ball, Var):
                raise inst_error()
            raise PrologError(self.copy(ball))
        elif k in BUILTINS:
            mark = len(self.trail)
            for ok in BUILTINS[k](self, args):
                if ok:
                    yield
                self.undo(mark)
        else:
            yield from self.call_user(g, k, depth)

    def collect(self, template: Term, goal: Term, depth: int) -> list[Term]:
        """findall's results (copies)."""
        results: list[Term] = []
        mark = len(self.trail)
        for _ in self.solve(goal, Frame(), depth):
            results.append(self.copy(template))
        self.undo(mark)
        return results

    def bagof(self, args: tuple[Term, ...], is_set: bool, depth: int) -> Iterator[None]:
        """bagof/setof with ^ and free-variable grouping (groups in standard order of the witness)."""
        template, goal, result = args
        inner = deref(goal)
        ex: set[int] = set()
        while isinstance(inner, Struct) and inner.name == "^" and len(inner.args) == 2:
            ex |= {id(v) for v in term_vars(inner.args[0])}
            inner = deref(inner.args[1])
        tv = {id(v) for v in term_vars(template)}
        witness = [v for v in term_vars(inner) if id(v) not in tv and id(v) not in ex]
        w = mklist(list(witness))
        pairs = self.collect(Struct("-", (w, template)), inner, depth)
        if not pairs:
            return
        items: list[tuple[Term, Term]] = []
        for p in pairs:
            assert isinstance(p, Struct)
            items.append((p.args[0], p.args[1]))

        def finish(xs: list[Term]) -> Term:
            return mklist(sort_terms(xs, True) if is_set else xs)

        mark = len(self.trail)
        if not witness:
            if self.unify(result, finish([b for _, b in items])):
                yield
            self.undo(mark)
            return
        from functools import cmp_to_key

        order = sorted(range(len(items)), key=cmp_to_key(lambda i, j: compare(items[i][0], items[j][0]) or (i - j)))
        groups: list[tuple[Term, list[Term]]] = []
        for i in order:
            key, val = items[i]
            if groups and variant(groups[-1][0], key):
                groups[-1][1].append(val)
            else:
                groups.append((key, [val]))
        for key, vals in groups:
            if self.unify(w, key) and self.unify(result, finish(vals)):
                yield
            self.undo(mark)

    def call_user(self, g: Term, k: tuple[str, int], depth: int) -> Iterator[None]:
        """Resolve against the clauses (snapshot: the logical update view)."""
        clauses = self.db.get(k)
        if clauses is None:
            raise err(Struct("existence_error", (Atom("procedure"), Struct("/", (Atom(k[0]), Int(k[1]))))))
        if depth + 1 > self.max_depth:
            self.depth_limited = True
            return
        frame = Frame()
        for c in list(clauses):
            mark = len(self.trail)
            r = self.copy(c)
            head, body = head_of(r), body_of(r)
            if self.unify(head, g):
                yield from self.solve(body, frame, depth + 1)
            self.undo(mark)
            if frame.cut:
                return

    def run(self, query: str, max_solutions: int = 1000) -> dict[str, object]:
        """Run a query: answers as [name, canonical text] pairs (fresh variables numbered `_V0`, ... in order)."""
        goal, names, vars_ = read_query(query)
        answers: list[list[list[str]]] = []
        status = "success"
        error: str | None = None
        try:
            for _ in self.solve(goal, Frame(), 0):
                numbering: dict[int, str] = {}
                pairs: list[list[str]] = []
                for n in names:
                    v = deref(vars_[n])
                    if v is vars_[n]:
                        continue
                    for fv in term_vars(v):
                        if id(fv) not in numbering:
                            numbering[id(fv)] = f"_V{len(numbering)}"
                    pairs.append([n, canonical(v, numbering)])
                answers.append(pairs)
                if len(answers) >= max_solutions:
                    break
        except PrologError as e:
            status = "error"
            ball = deref(e.ball)
            error = canonical(deref(ball.args[0]) if isinstance(ball, Struct) and ball.name == "error" else ball, {})
        except Budget:
            status = "budget"
        if status == "success" and not answers:
            status = "depth" if self.depth_limited else "failure"
        return {"answers": answers, "status": status, "error": error, "output": self.output}


def head_of(t: Term) -> Term:
    """A clause's head."""
    t = deref(t)
    if isinstance(t, Struct) and t.name == ":-" and len(t.args) == 2:
        return t.args[0]
    return t


def body_of(t: Term) -> Term:
    """A clause's body."""
    t = deref(t)
    if isinstance(t, Struct) and t.name == ":-" and len(t.args) == 2:
        return t.args[1]
    return TRUE


def term_vars(t: Term) -> list[Var]:
    """Distinct variables in order of first occurrence."""
    out: list[Var] = []
    seen: set[int] = set()

    def go(x: Term) -> None:
        x = deref(x)
        if isinstance(x, Var):
            if id(x) not in seen:
                seen.add(id(x))
                out.append(x)
        elif isinstance(x, Struct):
            for a in x.args:
                go(a)

    go(t)
    return out


def variant(a: Term, b: Term) -> bool:
    """Equal up to a bijective renaming of variables."""
    m: dict[int, int] = {}
    r: dict[int, int] = {}

    def go(x: Term, y: Term) -> bool:
        x, y = deref(x), deref(y)
        if isinstance(x, Var) and isinstance(y, Var):
            return not (m.setdefault(id(x), id(y)) != id(y) or r.setdefault(id(y), id(x)) != id(x))
        if isinstance(x, Struct) and isinstance(y, Struct):
            return (
                x.name == y.name
                and len(x.args) == len(y.args)
                and all(go(p, q) for p, q in zip(x.args, y.args, strict=True))
            )
        return x == y

    return go(a, b)


# ------------------------------------------------------------------------------------------------ built-ins

type BuiltinFn = Callable[[Machine, tuple[Term, ...]], Iterator[bool]]
BUILTINS: dict[tuple[str, int], BuiltinFn] = {}


def builtin(name: str, arity: int) -> Callable[[BuiltinFn], BuiltinFn]:
    """Register a built-in."""

    def register(f: BuiltinFn) -> BuiltinFn:
        BUILTINS[(name, arity)] = f
        return f

    return register


def det(f: Callable[[Machine, tuple[Term, ...]], bool]) -> BuiltinFn:
    """A deterministic built-in."""

    def run(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
        yield f(m, a)

    return run


def text_of(t: Term) -> str:
    """The text of an atomic term."""
    t = deref(t)
    if isinstance(t, Atom):
        return t.name
    if isinstance(t, Int):
        return str(t.value)
    if isinstance(t, Var):
        raise inst_error()
    raise type_error("atom", t)


def to_int(t: Term) -> int:
    """An integer argument."""
    t = deref(t)
    if isinstance(t, Int):
        return t.value
    if isinstance(t, Var):
        raise inst_error()
    raise type_error("integer", t)


def numeric_compare(test: Callable[[float, float], bool]) -> BuiltinFn:
    """An arithmetic comparison."""
    return det(lambda m, a: test(evaluate(a[0]).value, evaluate(a[1]).value))


def term_compare(test: Callable[[int], bool]) -> BuiltinFn:
    """A standard-order comparison."""
    return det(lambda m, a: test(compare(a[0], a[1])))


def not_unifiable(m: Machine, a: tuple[Term, ...]) -> bool:
    """\\=/2."""
    mark = len(m.trail)
    ok = m.unify(a[0], a[1])
    m.undo(mark)
    return not ok


BUILTINS[("=", 2)] = det(lambda m, a: m.unify(a[0], a[1]))
BUILTINS[("\\=", 2)] = det(not_unifiable)
BUILTINS[("==", 2)] = term_compare(lambda c: c == 0)
BUILTINS[("\\==", 2)] = term_compare(lambda c: c != 0)
BUILTINS[("@<", 2)] = term_compare(lambda c: c < 0)
BUILTINS[("@>", 2)] = term_compare(lambda c: c > 0)
BUILTINS[("@=<", 2)] = term_compare(lambda c: c <= 0)
BUILTINS[("@>=", 2)] = term_compare(lambda c: c >= 0)
BUILTINS[("=:=", 2)] = numeric_compare(lambda x, y: x == y)
BUILTINS[("=\\=", 2)] = numeric_compare(lambda x, y: x != y)
BUILTINS[("<", 2)] = numeric_compare(lambda x, y: x < y)
BUILTINS[(">", 2)] = numeric_compare(lambda x, y: x > y)
BUILTINS[("=<", 2)] = numeric_compare(lambda x, y: x <= y)
BUILTINS[(">=", 2)] = numeric_compare(lambda x, y: x >= y)
BUILTINS[("is", 2)] = det(lambda m, a: m.unify(a[0], evaluate(a[1])))
BUILTINS[("var", 1)] = det(lambda m, a: isinstance(deref(a[0]), Var))
BUILTINS[("nonvar", 1)] = det(lambda m, a: not isinstance(deref(a[0]), Var))
BUILTINS[("atom", 1)] = det(lambda m, a: isinstance(deref(a[0]), Atom))
BUILTINS[("number", 1)] = det(lambda m, a: isinstance(deref(a[0]), Int | Float))
BUILTINS[("integer", 1)] = det(lambda m, a: isinstance(deref(a[0]), Int))
BUILTINS[("atomic", 1)] = det(lambda m, a: isinstance(deref(a[0]), Atom | Int | Float))
BUILTINS[("compound", 1)] = det(lambda m, a: isinstance(deref(a[0]), Struct))
BUILTINS[("is_list", 1)] = det(lambda m, a: list_items(a[0]) is not None)
BUILTINS[("ground", 1)] = det(lambda m, a: not term_vars(a[0]))
BUILTINS[("copy_term", 2)] = det(lambda m, a: m.unify(a[1], m.copy(a[0])))
BUILTINS[("msort", 2)] = det(lambda m, a: m.unify(a[1], mklist(sort_terms(list_items(a[0]) or [], False))))
BUILTINS[("sort", 2)] = det(lambda m, a: m.unify(a[1], mklist(sort_terms(list_items(a[0]) or [], True))))
BUILTINS[("atom_length", 2)] = det(lambda m, a: m.unify(a[1], Int(len(text_of(a[0])))))
BUILTINS[("compare", 3)] = det(lambda m, a: m.unify(a[0], Atom({-1: "<", 0: "=", 1: ">"}[compare(a[1], a[2])])))


@builtin("between", 3)
def between(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """between/3."""
    lo, hi = to_int(a[0]), to_int(a[1])
    x = deref(a[2])
    if not isinstance(x, Var):
        yield lo <= to_int(x) <= hi
        return
    for i in range(lo, hi + 1):
        yield m.unify(x, Int(i))


@builtin("length", 2)
def length(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """length/2 (enumerating lengths of partial lists)."""
    count = 0
    t = deref(a[0])
    while isinstance(t, Struct) and t.name == "." and len(t.args) == 2:
        count += 1
        t = deref(t.args[1])
    if t == NIL:
        yield m.unify(a[1], Int(count))
        return
    if not isinstance(t, Var):
        yield False
        return
    n = deref(a[1])
    if isinstance(n, Int):
        yield n.value >= count and m.unify(t, mklist([m.fresh() for _ in range(n.value - count)]))
        return
    k = 0
    while True:
        yield m.unify(t, mklist([m.fresh() for _ in range(k)])) and m.unify(n, Int(count + k))
        k += 1


@builtin("functor", 3)
def functor(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """functor/3."""
    t = deref(a[0])
    if isinstance(t, Struct):
        yield m.unify(a[1], Atom(t.name)) and m.unify(a[2], Int(len(t.args)))
    elif not isinstance(t, Var):
        yield m.unify(a[1], t) and m.unify(a[2], Int(0))
    else:
        n, k = deref(a[1]), to_int(a[2])
        if k == 0:
            yield m.unify(t, n)
        elif isinstance(n, Atom):
            yield m.unify(t, Struct(n.name, tuple(m.fresh() for _ in range(k))))
        else:
            raise type_error("atomic", n)


@builtin("arg", 3)
def arg(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """arg/3."""
    t = deref(a[1])
    if not isinstance(t, Struct):
        raise type_error("compound", t)
    n = deref(a[0])
    if isinstance(n, Int):
        yield 1 <= n.value <= len(t.args) and m.unify(a[2], t.args[n.value - 1])
        return
    for i, x in enumerate(t.args):
        yield m.unify(n, Int(i + 1)) and m.unify(a[2], x)


@builtin("=..", 2)
def univ(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """=../2."""
    t = deref(a[0])
    if isinstance(t, Struct):
        yield m.unify(a[1], mklist([Atom(t.name), *t.args]))
    elif not isinstance(t, Var):
        yield m.unify(a[1], mklist([t]))
    else:
        items = list_items(a[1]) or []
        h = deref(items[0])
        if len(items) == 1:
            yield m.unify(t, h)
        else:
            assert isinstance(h, Atom)
            yield m.unify(t, Struct(h.name, tuple(items[1:])))


@builtin("atom_codes", 2)
def atom_codes(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """atom_codes/2."""
    t = deref(a[0])
    if not isinstance(t, Var):
        yield m.unify(a[1], mklist([Int(ord(c)) for c in text_of(t)]))
    else:
        yield m.unify(t, Atom("".join(chr(to_int(c)) for c in list_items(a[1]) or [])))


@builtin("atom_chars", 2)
def atom_chars(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """atom_chars/2."""
    t = deref(a[0])
    if not isinstance(t, Var):
        yield m.unify(a[1], mklist([Atom(c) for c in text_of(t)]))
    else:
        yield m.unify(t, Atom("".join(text_of(c) for c in list_items(a[1]) or [])))


@builtin("atom_concat", 3)
def atom_concat(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """atom_concat/3."""
    x, y = deref(a[0]), deref(a[1])
    if not isinstance(x, Var) and not isinstance(y, Var):
        yield m.unify(a[2], Atom(text_of(x) + text_of(y)))
        return
    whole = text_of(a[2])
    for i in range(len(whole) + 1):
        yield m.unify(x, Atom(whole[:i])) and m.unify(y, Atom(whole[i:]))


@builtin("keysort", 2)
def keysort(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """keysort/2 (stable)."""
    from functools import cmp_to_key

    pairs: list[Term] = [deref(p) for p in list_items(a[0]) or []]

    def key_of(p: Term) -> Term:
        assert isinstance(p, Struct)
        return p.args[0]

    pairs.sort(key=cmp_to_key(lambda p, q: compare(key_of(p), key_of(q))))
    yield m.unify(a[1], mklist(pairs))


@builtin("assertz", 1)
def assertz(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """assertz/1."""
    m.add(m.copy(a[0]), False)
    yield True


@builtin("asserta", 1)
def asserta(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """asserta/1."""
    m.add(m.copy(a[0]), True)
    yield True


BUILTINS[("assert", 1)] = assertz


@builtin("retract", 1)
def retract(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """retract/1 (nondeterministic, over a snapshot)."""
    t = deref(a[0])
    head, body = head_of(t), body_of(t)
    k = m.key(head)
    for c in list(m.db.get(k, [])):
        if c not in m.db.get(k, []):
            continue
        r = m.copy(c)
        if m.unify(head, head_of(r)) and m.unify(body, body_of(r)):
            m.db[k] = [x for x in m.db[k] if x is not c]
            yield True
        else:
            yield False


@builtin("write", 1)
def write(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """write/1."""
    m.output += write_text(a[0], False)
    yield True


@builtin("nl", 0)
def nl(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """nl/0."""
    m.output += "\n"
    yield True


@builtin("format", 2)
def format_(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """format/2 with ~w, ~a, ~d and ~n."""
    fmt = text_of(a[0])
    args = list_items(a[1])
    items = args if args is not None else [a[1]]
    out = ""
    i = 0
    k = 0
    while i < len(fmt):
        c = fmt[i]
        if c == "~":
            d = fmt[i + 1]
            if d == "n":
                out += "\n"
            elif d == "~":
                out += "~"
            elif d in "wad":
                x = items[k]
                k += 1
                out += write_text(x, False) if d == "w" else text_of(x)
            i += 2
            continue
        out += c
        i += 1
    m.output += out
    yield True


@builtin("succ", 2)
def succ(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """succ/2."""
    x = deref(a[0])
    if isinstance(x, Int):
        yield m.unify(a[1], Int(x.value + 1))
    else:
        n = to_int(a[1])
        yield n > 0 and m.unify(x, Int(n - 1))


@builtin("plus", 3)
def plus(m: Machine, a: tuple[Term, ...]) -> Iterator[bool]:
    """plus/3."""
    x, y, z = deref(a[0]), deref(a[1]), deref(a[2])
    if isinstance(x, Int) and isinstance(y, Int):
        yield m.unify(z, Int(x.value + y.value))
    elif isinstance(x, Int):
        yield m.unify(y, Int(to_int(z) - x.value))
    else:
        yield m.unify(x, Int(to_int(z) - to_int(y)))


def run_query(program: str, query: str, max_solutions: int = 1000, max_depth: int = 10_000) -> dict[str, object]:
    """Consult a program and run one query."""
    return Machine(program, max_depth=max_depth).run(query, max_solutions)
