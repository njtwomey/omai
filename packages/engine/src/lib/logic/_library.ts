// The standard library of the resolution engine, written in Prolog. Definitions follow SWI-Prolog's library(lists),
// library(apply) and library(aggregate) so that answers come in the same order. Helpers are named with a `$` prefix.
export const LIBRARY_SOURCE = `
append([], L, L).
append([H|T], L, [H|R]) :- append(T, L, R).

append([], []).
append([L|Ls], As) :- append(L, Ws, As), append(Ls, Ws).

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

nextto(X, Y, [X,Y|_]).
nextto(X, Y, [_|T]) :- nextto(X, Y, T).

select(X, [X|T], T).
select(X, [H|T], [H|R]) :- select(X, T, R).

selectchk(X, L, R) :- select(X, L, R), !.

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

partition(_, [], [], []).
partition(P, [H|T], I, E) :- ( call(P, H) -> I = [H|I1], E = E1 ; I = I1, E = [H|E1] ), partition(P, T, I1, E1).

maplist(_, []).
maplist(P, [A|As]) :- call(P, A), maplist(P, As).
maplist(_, [], []).
maplist(P, [A|As], [B|Bs]) :- call(P, A, B), maplist(P, As, Bs).
maplist(_, [], [], []).
maplist(P, [A|As], [B|Bs], [C|Cs]) :- call(P, A, B, C), maplist(P, As, Bs, Cs).
maplist(_, [], [], [], []).
maplist(P, [A|As], [B|Bs], [C|Cs], [D|Ds]) :- call(P, A, B, C, D), maplist(P, As, Bs, Cs, Ds).

foldl(G, L, V0, V) :- '$foldl'(L, G, V0, V).
'$foldl'([], _, V, V).
'$foldl'([X|Xs], G, V0, V) :- call(G, X, V0, V1), '$foldl'(Xs, G, V1, V).
foldl(G, L1, L2, V0, V) :- '$foldl'(L1, L2, G, V0, V).
'$foldl'([], [], _, V, V).
'$foldl'([X|Xs], [Y|Ys], G, V0, V) :- call(G, X, Y, V0, V1), '$foldl'(Xs, Ys, G, V1, V).
foldl(G, L1, L2, L3, V0, V) :- '$foldl'(L1, L2, L3, G, V0, V).
'$foldl'([], [], [], _, V, V).
'$foldl'([X|Xs], [Y|Ys], [Z|Zs], G, V0, V) :- call(G, X, Y, Z, V0, V1), '$foldl'(Xs, Ys, Zs, G, V1, V).

sum_list(Xs, Sum) :- '$sum_list'(Xs, 0, Sum).
'$sum_list'([], Sum, Sum).
'$sum_list'([X|Xs], Sum0, Sum) :- Sum1 is Sum0 + X, '$sum_list'(Xs, Sum1, Sum).
sumlist(Xs, Sum) :- sum_list(Xs, Sum).

max_list([H|T], Max) :- '$max_list'(T, H, Max).
'$max_list'([], Max, Max).
'$max_list'([H|T], Max0, Max) :- Max1 is max(H, Max0), '$max_list'(T, Max1, Max).
min_list([H|T], Min) :- '$min_list'(T, H, Min).
'$min_list'([], Min, Min).
'$min_list'([H|T], Min0, Min) :- Min1 is min(H, Min0), '$min_list'(T, Min1, Min).

max_member(Max, [H|T]) :- '$max_member'(T, H, Max).
'$max_member'([], Max, Max).
'$max_member'([H|T], Max0, Max) :- ( H @=< Max0 -> Max1 = Max0 ; Max1 = H ), '$max_member'(T, Max1, Max).
min_member(Min, [H|T]) :- '$min_member'(T, H, Min).
'$min_member'([], Min, Min).
'$min_member'([H|T], Min0, Min) :- ( H @>= Min0 -> Min1 = Min0 ; Min1 = H ), '$min_member'(T, Min1, Min).

numlist(L, H, R) :- L =< H, '$numlist'(L, H, R).
'$numlist'(H, H, [H]) :- !.
'$numlist'(L, H, [L|T]) :- L1 is L + 1, '$numlist'(L1, H, T).

list_to_set(L, S) :- '$lts'(L, [], S).
'$lts'([], _, []).
'$lts'([H|T], Seen, R) :- ( '$memberchk_eq'(H, Seen) -> R = R1 ; R = [H|R1] ), '$lts'(T, [H|Seen], R1).
'$memberchk_eq'(X, [Y|Ys]) :- ( X == Y -> true ; '$memberchk_eq'(X, Ys) ).

permutation([], []).
permutation(L, [X|P]) :- select(X, L, R), permutation(R, P).

flatten(List, Flat) :- '$flatten'(List, [], Flat0), !, Flat = Flat0.
'$flatten'(Var, Tl, [Var|Tl]) :- var(Var), !.
'$flatten'([], Tl, Tl) :- !.
'$flatten'([Hd|Tl], Tail, List) :- !, '$flatten'(Hd, FlatHeadTail, List), '$flatten'(Tl, Tail, FlatHeadTail).
'$flatten'(NonList, Tl, [NonList|Tl]).

pairs_keys_values([], [], []).
pairs_keys_values([K-V|T], [K|Ks], [V|Vs]) :- pairs_keys_values(T, Ks, Vs).
pairs_keys([], []).
pairs_keys([K-_|T], [K|Ks]) :- pairs_keys(T, Ks).
pairs_values([], []).
pairs_values([_-V|T], [V|Vs]) :- pairs_values(T, Vs).

predsort(P, L, Sorted) :- length(L, N), '$predsort'(P, N, L, _, Sorted1), !, Sorted = Sorted1.
'$predsort'(P, 2, [X1,X2|L], L, R) :- !, call(P, Delta, X1, X2), '$sort2'(Delta, X1, X2, R).
'$predsort'(_, 1, [X|L], L, [X]) :- !.
'$predsort'(_, 0, L, L, []) :- !.
'$predsort'(P, N, L1, L3, R) :-
    N1 is N // 2, plus(N1, N2, N),
    '$predsort'(P, N1, L1, L2, R1), '$predsort'(P, N2, L2, L3, R2),
    '$predmerge'(P, R1, R2, R).
'$sort2'(<, X1, X2, [X1,X2]).
'$sort2'(=, X1, _, [X1]).
'$sort2'(>, X1, X2, [X2,X1]).
'$predmerge'(_, [], R, R) :- !.
'$predmerge'(_, R, [], R) :- !.
'$predmerge'(P, [H1|T1], [H2|T2], Result) :-
    call(P, Delta, H1, H2), !, '$predmerge_'(Delta, P, H1, H2, T1, T2, Result).
'$predmerge_'(<, P, H1, H2, T1, T2, [H1|R]) :- '$predmerge'(P, T1, [H2|T2], R).
'$predmerge_'(=, P, H1, _, T1, T2, [H1|R]) :- '$predmerge'(P, T1, T2, R).
'$predmerge_'(>, P, H1, H2, T1, T2, [H2|R]) :- '$predmerge'(P, [H1|T1], T2, R).

once(G) :- call(G), !.
ignore(G) :- ( call(G) -> true ; true ).
forall(C, A) :- \\+ ( call(C), \\+ call(A) ).

aggregate_all(count, G, C) :- !, findall(x, G, L), length(L, C).
aggregate_all(sum(E), G, S) :- !, findall(E, G, L), sum_list(L, S).
aggregate_all(max(E), G, M) :- !, findall(E, G, L), L = [_|_], max_list(L, M).
aggregate_all(min(E), G, M) :- !, findall(E, G, L), L = [_|_], min_list(L, M).
aggregate_all(bag(E), G, L) :- !, findall(E, G, L).
aggregate_all(set(E), G, S) :- !, findall(E, G, L), sort(L, S).

phrase(G, L) :- phrase(G, L, []).
phrase(G, L, R) :- call(G, L, R).
`
