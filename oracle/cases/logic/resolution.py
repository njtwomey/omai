"""SLD resolution against a reference Prolog interpreter (oracle/cases/logic/_prolog.py, a typed generator-based port
standing in for SWI-Prolog, which uv cannot install): answers in order, statuses, ISO error terms and output for
textbook programs (family relations, list processing, cut, negation, arithmetic, n-queens, all-solutions predicates,
exceptions, the dynamic database and depth limits). Answers are written canonically (no operators), with fresh
variables numbered `_V0`, `_V1`, ... by first occurrence."""

from cases.logic._prolog import run_query
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"reference-prolog": "generator-based port (oracle/cases/logic/_prolog.py)"}

FAMILY = """
parent(tom, bob). parent(tom, liz). parent(bob, ann). parent(bob, pat). parent(pat, jim).
male(tom). male(bob). male(jim). female(liz). female(ann). female(pat).
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
sibling(X, Y) :- parent(P, X), parent(P, Y), X \\= Y.
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
"""

LISTS = """
len([], 0).
len([_|T], N) :- len(T, M), N is M + 1.
sumsq([], 0).
sumsq([X|Xs], S) :- sumsq(Xs, S0), S is S0 + X * X.
evens([], []).
evens([X|Xs], [X|Ys]) :- 0 is X mod 2, !, evens(Xs, Ys).
evens([_|Xs], Ys) :- evens(Xs, Ys).
double(X, Y) :- Y is 2 * X.
add(X, Y, Z) :- Z is X + Y.
big(X) :- X > 2.
"""

CONTROL = """
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
first(X) :- member(X, [a, b, c]), !.
classify(X, C) :- ( X < 0 -> C = negative ; X =:= 0 -> C = zero ; C = positive ).
not_member(X, L) :- \\+ member(X, L).
t(1). t(2). t(3).
cut_in_call(X) :- call((t(X), !)).
cut_in_call(9).
neg(X) :- t(X), \\+ X = 2.
"""

ARITH = """
fact(0, 1) :- !.
fact(N, F) :- N > 0, N1 is N - 1, fact(N1, F1), F is N * F1.
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, A is N - 1, B is N - 2, fib(A, FA), fib(B, FB), F is FA + FB.
gcd(X, 0, X) :- !.
gcd(X, Y, G) :- Z is X mod Y, gcd(Y, Z, G).
collatz(1, [1]) :- !.
collatz(N, [N|T]) :- ( 0 is N mod 2 -> M is N // 2 ; M is 3 * N + 1 ), collatz(M, T).
"""

QUEENS = """
queens(N, Qs) :- numlist(1, N, Ns), permutation(Ns, Qs), safe(Qs).
safe([]).
safe([Q|Qs]) :- no_attack(Q, Qs, 1), safe(Qs).
no_attack(_, [], _).
no_attack(Q, [Q1|Qs], D) :- Q =\\= Q1 + D, Q =\\= Q1 - D, D1 is D + 1, no_attack(Q, Qs, D1).
"""

AGES = "age(peter, 7). age(ann, 11). age(pat, 8). age(tom, 5). age(mike, 11). likes(ann, X) :- age(X, A), A > 7."

DYNAMIC = """
:- dynamic(counter/1).
:- dynamic(item/1).
counter(0).
item(a). item(b). item(c).
incr :- retract(counter(N)), N1 is N + 1, assertz(counter(N1)).
"""

NAT = "nat(0). nat(s(X)) :- nat(X). add(0, Y, Y). add(s(X), Y, s(Z)) :- add(X, Y, Z)."

HANOI = """
hanoi(0, _, _, _) :- !.
hanoi(N, A, B, C) :- M is N - 1, hanoi(M, A, C, B), format('~w->~w ', [A, B]), hanoi(M, C, B, A).
"""

PROGRAMS: list[tuple[str, str, list[str]]] = [
    (
        "family",
        FAMILY,
        [
            "ancestor(tom, D)",
            "ancestor(A, jim)",
            "grandparent(G, ann)",
            "sibling(ann, S)",
            "father(F, C)",
            "mother(M, jim)",
            "findall(X-Y, grandparent(X, Y), L)",
        ],
    ),
    (
        "lists",
        LISTS,
        [
            "append(X, Y, [1,2,3])",
            "append([a], T, L)",
            "member(X, [a, b, c])",
            "reverse([1,2,3,4], R)",
            "len([a, b, c], N)",
            "sumsq([1,2,3], S)",
            "evens([1,2,3,4,5,6], E)",
            "select(X, [a,b,c], R)",
            "permutation([1,2,3], P)",
            "nth0(I, [a,b,c], E)",
            "nth1(2, [a,b,c], E)",
            "last([1,2,3], L)",
            "maplist(double, [1,2,3], L)",
            "foldl(add, [1,2,3], 0, S)",
            "include(big, [1,2,3,4], I), exclude(big, [1,2,3,4], E)",
            "msort([b, 2, a, 1.0, f(x), 1], L)",
            "sort([c, a, b, a, c], L)",
            "keysort([b-1, a-2, b-0, a-1], L)",
            "subtract([1,2,3,4], [2,4], A), union([1,2], [2,3], B), intersection([1,2,3], [3,1], C)",
            "delete([a, b, a, c], a, D)",
            "length(L, 2)",
            "length(L, N)",
            "sum_list([1, 2, 3.5], S), max_list([3, 1, 4], M), min_list([3, 1, 4], N)",
            "memberchk(X, [a, b])",
        ],
    ),
    (
        "control",
        CONTROL,
        [
            "max(3, 2, M)",
            "max(1, 2, M)",
            "first(X)",
            "classify(-5, A), classify(0, B), classify(7, C)",
            "not_member(d, [a, b])",
            "not_member(a, [a, b])",
            "cut_in_call(X)",
            "neg(X)",
            "( t(X), X > 1 -> Y = yes ; Y = no )",
            "( t(X) ; X = 4 )",
            "forall(t(X), X > 0)",
            "once(t(X))",
            "ignore(fail)",
            "\\+ t(5)",
            "t(X), !, t(Y)",
        ],
    ),
    (
        "arithmetic",
        ARITH,
        [
            "fact(10, F)",
            "fib(15, F)",
            "gcd(1071, 462, G)",
            "collatz(6, L)",
            "X is 7 / 2, Y is 7 // 2, Z is -7 // 2, W is 8 / 4",
            "X is -7 mod 3, Y is -7 rem 3, Z is 7 mod -3",
            "X is 2 ** 10, Y is 2 ** -1, Z is 2 ^ 5, W is 2.0 * 3",
            "X is max(3, 4.5), Y is min(2, 1), Z is abs(-4), W is truncate(3.7)",
            "1 =:= 1.0",
            "1 == 1.0",
            "X = 1 + 2, Y is X * 3",
            "between(1, 4, X), 0 is X mod 2",
            "succ(X, 5), succ(3, Y), plus(2, Z, 9)",
        ],
    ),
    ("queens", QUEENS, ["queens(4, Qs)", "queens(5, Qs)", "findall(Q, queens(6, Q), L), length(L, N)"]),
    (
        "all-solutions",
        AGES,
        [
            "findall(N, age(N, _), L)",
            "bagof(N, age(N, A), L)",
            "setof(N, A^age(N, A), L)",
            "setof(A-N, age(N, A), L)",
            "bagof(N, age(N, 99), L)",
            "findall(X, likes(ann, X), L)",
            "aggregate_all(count, age(_, _), C)",
            "aggregate_all(sum(A), age(_, A), S), aggregate_all(max(A), age(_, A), M)",
            "aggregate_all(set(A), age(_, A), S)",
            "setof(X-Y, member(X-Y, [b-1, a-2, a-1, b-1]), L)",
        ],
    ),
    (
        "exceptions",
        "p(1). q(X) :- X > 0.",
        [
            "catch(X is Y + 1, error(E, _), true)",
            "catch(X is foo + 1, error(E, _), true)",
            "catch(undefined_thing(1), error(E, _), true)",
            "catch(throw(ball(1)), ball(X), true)",
            "catch((member(X, [1,2,3]), X > 1, throw(found(X))), found(Y), true)",
            "catch(X is 1 / 0, error(E, _), true)",
            "catch(atom_length(X, L), error(E, _), true)",
            "catch(q(a), error(E, _), true)",
            "X is 1 / 0",
            "undefined_thing",
            "catch(p(X), _, true)",
            "catch(throw(x), y, true)",
        ],
    ),
    (
        "dynamic",
        DYNAMIC,
        [
            "incr, incr, counter(X)",
            "retract(item(b)), findall(X, item(X), L)",
            "item(X), assertz(item(d))",
            "asserta(item(z)), findall(X, item(X), L)",
            "retract(item(X))",
            "assertz(new(1)), assertz(new(2)), findall(X, new(X), L)",
        ],
    ),
    (
        "terms",
        "",
        [
            "functor(foo(a, b, c), N, A)",
            "functor(T, point, 3)",
            "arg(2, f(a, b, c), X)",
            "arg(N, f(a, b), X)",
            "f(a, b) =.. L",
            "T =.. [g, 1, 2]",
            "copy_term(f(X, Y, X), C)",
            "atom_codes(hello, C)",
            "atom_codes(A, [104, 105])",
            "atom_chars(abc, C)",
            "atom_length(prolog, N)",
            "atom_concat(abc, def, X)",
            "atom_concat(X, Y, abc)",
            'X = "hi"',
            "compare(O, 1, a), compare(P, f(b), f(a)), compare(Q, x, x)",
            "X = f(Y), Y = g(Z), Z = 1",
            "f(X, b) = f(a, Y)",
            "f(X, X) = f(a, b)",
            "a @< b, f(a) @> b, 1.0 @< 1",
            "X == X, \\+ X == Y",
            "atom(foo), \\+ atom(1), number(1.5), integer(3), \\+ integer(3.0), compound(f(x)), var(_), atomic(a)",
            "is_list([a]), \\+ is_list([a|_]), ground(f(a)), \\+ ground(f(_))",
        ],
    ),
    ("output", HANOI, ["hanoi(3, a, b, c)", "write(hello), nl, write([1, 2]), format('~a and ~d~n', [x, 42])"]),
]

DEPTH_QUERIES: list[tuple[str, str, int]] = [
    (NAT, "nat(X)", 4),
    (NAT, "nat(X), fail", 6),
    (NAT, "add(X, Y, s(s(0)))", 10),
    ("p(X) :- p(X).", "p(1)", 50),
    (FAMILY, "ancestor(tom, X)", 2),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, program, queries in PROGRAMS:
        for q in queries:
            r = run_query(program, q, max_solutions=50)
            out.append(case(f"{name}/{q}", "solve", [], r["answers"], program=program, query=q, result=r))
    for program, q, depth in DEPTH_QUERIES:
        r = run_query(program, q, max_solutions=50, max_depth=depth)
        out.append(
            case(f"depth/{depth}/{q}", "solveDepth", [], r["answers"], program=program, query=q, depth=depth, result=r)
        )
    return out
