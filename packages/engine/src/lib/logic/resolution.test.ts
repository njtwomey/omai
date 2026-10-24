import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, validateAgainstSchema } from '../trace/index.ts'
import { consult, prove, sldAlgorithm, sldTree, solve, termToString } from './index.ts'

const texts = (program: string, query: string, options = {}): string[] =>
  solve(program, query, options).answers.map((a) => a.text)

describe('SLD resolution', () => {
  it('answers in clause order with backtracking', () => {
    const family = `
      parent(tom, bob). parent(tom, liz). parent(bob, ann). parent(bob, pat). parent(pat, jim).
      ancestor(X, Y) :- parent(X, Y).
      ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).`
    expect(texts(family, 'ancestor(tom, D)')).toEqual(['D = bob', 'D = liz', 'D = ann', 'D = pat', 'D = jim'])
    expect(texts(family, 'ancestor(A, jim)')).toEqual(['A = pat', 'A = tom', 'A = bob'])
  })

  it('cut prunes clause alternatives and is local to call/1', () => {
    const p = `
      max(X, Y, X) :- X >= Y, !.
      max(_, Y, Y).
      t(X) :- member(X, [1,2,3]), !.
      u(X) :- call((member(X, [1,2,3]), !)).
      u(9).`
    expect(texts(p, 'max(3, 2, M)')).toEqual(['M = 3'])
    expect(texts(p, 'max(1, 2, M)')).toEqual(['M = 2'])
    expect(texts(p, 't(X)')).toEqual(['X = 1'])
    expect(texts(p, 'u(X)')).toEqual(['X = 1', 'X = 9'])
  })

  it('if-then-else, soft cut, negation and disjunction', () => {
    expect(texts('', '( member(X, [1,2,3]), X > 1 -> Y = big ; Y = small )')).toEqual(['X = 2, Y = big'])
    expect(texts('', '( fail -> Y = a ; Y = b )')).toEqual(['Y = b'])
    expect(texts('', '( member(X, [1,2]) *-> true ; X = none )')).toEqual(['X = 1', 'X = 2'])
    expect(texts('', '( fail *-> true ; X = none )')).toEqual(['X = none'])
    expect(texts('', '\\+ member(4, [1,2])')).toEqual(['true'])
    expect(texts('', '( X = 1 ; X = 2 ; X = 3 )')).toEqual(['X = 1', 'X = 2', 'X = 3'])
    expect(texts('', '( true -> fail ; true )')).toEqual([])
  })

  it('collects solutions with findall, bagof and setof', () => {
    const p = 'age(peter, 7). age(ann, 11). age(pat, 8). age(tom, 5). age(mike, 11).'
    expect(texts(p, 'findall(N, age(N, _), L)')).toEqual(['L = [peter,ann,pat,tom,mike]'])
    expect(texts(p, 'setof(A-N, age(N, A), L)')).toEqual(['L = [5-tom,7-peter,8-pat,11-ann,11-mike]'])
    expect(texts(p, 'bagof(N, age(N, A), L)')).toEqual([
      'A = 5, L = [tom]',
      'A = 7, L = [peter]',
      'A = 8, L = [pat]',
      'A = 11, L = [ann,mike]',
    ])
    expect(texts(p, 'setof(N, A^age(N, A), L)')).toEqual(['L = [ann,mike,pat,peter,tom]'])
    expect(texts(p, 'bagof(N, age(N, 99), L)')).toEqual([])
    expect(texts(p, 'findall(X, member(X, [a,b]), L, [c])')).toEqual(['L = [a,b,c]'])
    expect(texts(p, 'aggregate_all(count, age(_, _), C), aggregate_all(max(A), age(_, A), M)')).toEqual([
      'C = 5, M = 11',
    ])
  })

  it('raises and catches ISO errors', () => {
    expect(texts('', 'catch(X is Y + 1, error(E, _), true)')).toEqual(['E = instantiation_error'])
    expect(texts('', 'catch(atom_length(1, 2, 3), error(E, _), true)')[0]).toMatch(/existence_error/)
    expect(texts('', 'catch(throw(my), Ball, true)')).toEqual(['Ball = my'])
    expect(texts('', 'catch((member(X, [1,2,3]), X > 1, throw(found(X))), found(Y), true)')).toEqual(['Y = 2'])
    const r = solve('', 'undefined_pred(1)')
    expect(r.status).toBe('error')
    expect(termToString(r.error ?? { kind: 'atom', name: '' })).toMatch(
      /existence_error\(procedure,undefined_pred\/1\)/,
    )
    expect(solve('', 'undefined_pred(1)', { unknown: 'fail' }).status).toBe('failure')
    // A catch is no longer active after its goal has exited.
    expect(solve('', 'catch(true, _, true), throw(oops)').status).toBe('error')
    // ... but is again on backtracking into the goal.
    expect(texts('', 'catch(member(X, [1,2]), E, (X = caught(E))), X > 1, true')).toEqual(['X = 2'])
  })

  it('arithmetic follows SWI-Prolog defaults', () => {
    expect(texts('', 'X is 2 ** 10, Y is 2 ** -1, Z is 2.0 * 3')).toEqual(['X = 1024, Y = 0.5, Z = 6.0'])
    expect(texts('', 'X is -7 // 2, Y is -7 mod 2, Z is -7 rem 2, W is 7 / 7')).toEqual([
      'X = -3, Y = 1, Z = -1, W = 1',
    ])
    expect(texts('', 'X is max(1, 2.5), Y is abs(-3), Z is truncate(3.7), W is 17 xor 5')).toEqual([
      'X = 2.5, Y = 3, Z = 3, W = 20',
    ])
    expect(prove('', '1 =:= 1.0')).toBe(true)
    expect(prove('', '1 == 1.0')).toBe(false)
    expect(texts('', 'catch(X is 1 / 0, error(E, _), true)')).toEqual(['E = evaluation_error(zero_divisor)'])
  })

  it('inspects and builds terms', () => {
    expect(texts('', 'functor(foo(a, b), N, A), T =.. [bar, 1, 2], arg(2, foo(a, b), X)')).toEqual([
      'N = foo, A = 2, T = bar(1,2), X = b',
    ])
    expect(texts('', 'functor(T, f, 2), copy_term(f(X, Y, X), C)')[0]).toMatch(
      /^T = f\(_\d+,_\d+\), C = f\((_\d+),_\d+,\1\)$/,
    )
    expect(texts('', 'arg(N, f(a, b), b)')).toEqual(['N = 2'])
    expect(texts('', 'length(L, 2)')[0]).toMatch(/^L = \[_\d+,_\d+\]$/)
    expect(texts('', 'length(L, N)', { maxSolutions: 3 }).length).toBe(3)
    expect(texts('', 'atom_codes(abc, C), atom_chars(X, [h, i]), atom_length(hello, N)')).toEqual([
      'C = [97,98,99], X = hi, N = 5',
    ])
    expect(texts('', 'atom_concat(X, Y, ab)')).toEqual(["X = '', Y = ab", 'X = a, Y = b', "X = ab, Y = ''"])
    expect(texts('', 'sub_atom(hello, B, 2, A, ll)')).toEqual(['B = 2, A = 1'])
    expect(texts('', "atomic_list_concat([a, b, c], '-', X), atomic_list_concat(L, ',', 'x,y')")).toEqual([
      "X = 'a-b-c', L = [x,y]",
    ])
    expect(texts('', "atom_number(A, 12), atom_number('3.5', N)")).toEqual(["A = '12', N = 3.5"])
    expect(texts('', 'term_to_atom(f(X, "a"), A), term_to_atom(T, \'g(Y, 1 + 2)\')')[0]).toMatch(
      /^A = 'f\(X,\[97\]\)', T = g\(_\d+,1\+2\)$/,
    )
    expect(texts('', 'X = "ab"')).toEqual(['X = [97,98]'])
  })

  it('sorts in the standard order of terms', () => {
    expect(texts('', 'msort([b, 1, a, 2.0, f(x), Z, a], L)')[0]).toMatch(/^L = \[Z,1,2\.0,a,a,b,f\(x\)\]$/)
    expect(texts('', 'sort([c, a, b, a], L), sort(0, @>=, [1, 3, 2, 3], D)')).toEqual(['L = [a,b,c], D = [3,3,2,1]'])
    expect(texts('', 'keysort([b-1, a-2, b-0, a-1], L)')).toEqual(['L = [a-2,a-1,b-1,b-0]'])
    expect(texts('cmp(O, A, B) :- compare(O, B, A).', 'predsort(cmp, [1, 3, 2, 3], L)')).toEqual(['L = [3,2,1]'])
  })

  it('uses the standard library', () => {
    expect(texts('', 'reverse([1,2,3], R), nth0(1, [a,b,c], E0), nth1(1, [a,b,c], E1), last([a,b,c], L)')).toEqual([
      'R = [3,2,1], E0 = b, E1 = a, L = c',
    ])
    expect(texts('', 'permutation([1,2,3], P)')).toEqual([
      'P = [1,2,3]',
      'P = [1,3,2]',
      'P = [2,1,3]',
      'P = [2,3,1]',
      'P = [3,1,2]',
      'P = [3,2,1]',
    ])
    expect(texts('', 'msort([c,a], X), sum_list([1, 2.5], S), max_list([3, 1, 4], M), numlist(1, 4, N)')).toEqual([
      'X = [a,c], S = 3.5, M = 4, N = [1,2,3,4]',
    ])
    expect(
      texts('p(X) :- X > 1.', 'include(p, [0, 2, 3, 1], I), exclude(p, [0, 2, 3, 1], E), partition(p, [0, 2], A, B)'),
    ).toEqual(['I = [2,3], E = [0,1], A = [2], B = [0]'])
    expect(texts('add(X, Y, Z) :- Z is X + Y.', 'maplist(add(1), [1, 2], L), foldl(add, [1, 2, 3], 0, S)')).toEqual([
      'L = [2,3], S = 6',
    ])
    expect(texts('', 'flatten([1, [2, [3, 4]], [], 5], F), list_to_set([a, b, a, c, b], S)')).toEqual([
      'F = [1,2,3,4,5], S = [a,b,c]',
    ])
    expect(
      texts(
        '',
        'subtract([1,2,3,4], [2,4], A), intersection([1,2,3], [3,1], B), union([1,2], [2,3], C), delete([a,b,a,c], a, D)',
      ),
    ).toEqual(['A = [1,3], B = [1,3], C = [1,2,3], D = [b,c]'])
    expect(texts('', 'between(1, 3, X)')).toEqual(['X = 1', 'X = 2', 'X = 3'])
    expect(texts('', 'forall(member(X, [1,2]), X > 0), once(member(Y, [a, b]))')).toEqual(['Y = a'])
    expect(texts('', 'succ(X, 4), succ(3, Y), plus(2, Z, 5)')).toEqual(['X = 3, Y = 4, Z = 3'])
  })

  it('modifies the database with the logical update view', () => {
    const r = solve(
      ':- dynamic(counter/1). counter(0).',
      'retract(counter(N)), N1 is N + 1, assertz(counter(N1)), counter(X)',
    )
    expect(r.answers.map((a) => a.text)).toEqual(['N = 0, N1 = 1, X = 1'])
    expect(r.database.predicates.get('counter/1')?.length).toBe(1)
    // A running call does not see clauses added after it started.
    expect(texts(':- dynamic(q/1). q(1).', 'q(X), assertz(q(2))')).toEqual(['X = 1'])
    expect(texts(':- dynamic(q/1). q(1). q(2). q(3).', 'retractall(q(2)), findall(X, q(X), L)')).toEqual(['L = [1,3]'])
    expect(texts('p(1) :- true. p(2) :- q.', 'clause(p(X), B)')).toEqual(['X = 1, B = true', 'X = 2, B = q'])
    expect(solve('', 'assertz(foo(1)), foo(X)').answers.map((a) => a.text)).toEqual(['X = 1'])
    expect(solve('', 'catch(assertz((atom(_) :- true)), error(E, _), true)').answers[0]?.text).toMatch(
      /permission_error/,
    )
  })

  it('writes output', () => {
    const r = solve(
      '',
      "write(hello), nl, print('A b'), format(' ~w and ~a: ~d~n', [f(x), 'q q', 42]), format('~2f|~e', [3.14159, 2.5])",
    )
    expect(r.output).toBe("hello\n'A b' f(x) and q q: 42\n3.14|2.500000e+00")
    expect(solve('', "writeq([a, 'B', 1+2, -(1), - a, f(- 1)]), write_canonical(1+2)").output).toBe(
      "[a,'B',1+2,- 1,-a,f(- 1)]+(1,2)",
    )
  })

  it('translates definite clause grammars', () => {
    const grammar = `
      greeting --> [hello], name.
      name --> [world].
      name --> [prolog].
      digits([D|T]) --> digit(D), digits(T).
      digits([D]) --> digit(D).
      digit(D) --> [D], { code_type_digit(D) }.
      code_type_digit(D) :- D >= 0'0, D =< 0'9.`
    expect(texts(grammar, 'phrase(greeting, [hello, prolog])')).toEqual(['true'])
    expect(texts(grammar, 'phrase(greeting, [hello, X])')).toEqual(['X = world', 'X = prolog'])
    expect(texts(grammar, 'phrase(digits(D), "12", R)')).toEqual(['D = [49,50], R = []', 'D = [49], R = [50]'])
  })

  it('solves n-queens and counts its solutions', () => {
    const queens = `
      queens(N, Qs) :- numlist(1, N, Ns), permutation(Ns, Qs), safe(Qs).
      safe([]).
      safe([Q|Qs]) :- no_attack(Q, Qs, 1), safe(Qs).
      no_attack(_, [], _).
      no_attack(Q, [Q1|Qs], D) :- Q =\\= Q1 + D, Q =\\= Q1 - D, D1 is D + 1, no_attack(Q, Qs, D1).`
    expect(texts(queens, 'queens(4, Qs)')).toEqual(['Qs = [2,4,1,3]', 'Qs = [3,1,4,2]'])
    expect(solve(queens, 'queens(6, Qs)').answers.length).toBe(4)
  })

  it('reports budgets, depth limits and the occurs check', () => {
    expect(solve('p :- p.', 'p', { maxSteps: 500 }).status).toBe('budget')
    const nat = 'nat(0). nat(s(X)) :- nat(X).'
    const r = solve(nat, 'nat(X)', { maxDepth: 3 })
    expect(r.answers.map((a) => a.text)).toEqual(['X = 0', 'X = s(0)', 'X = s(s(0))'])
    expect([r.status, r.depthLimited, r.exhausted]).toEqual(['success', true, true])
    expect(texts('', 'X = f(X)', { occursCheck: true })).toEqual([])
    expect(texts('', 'unify_with_occurs_check(X, f(X))')).toEqual([])
    expect(solve(nat, 'nat(X)', { maxSolutions: 2 }).exhausted).toBe(false)
  })

  it('records an SLD tree and a Byrd-box trace', () => {
    const r = solve('p(1). p(2). q(2).', 'p(X), q(X)', { recordTree: true })
    const tree = sldTree(r)
    expect(tree.edges.map((e) => e.label)).toEqual(['p/1 #1', 'q/1: no clauses', 'p/1 #2', 'q/1 #1'])
    expect(tree.nodes.filter((n) => n.role === 'output').length).toBe(1)
    const rec = recorder()
    solve('p(1). p(2). q(2).', 'p(X), q(X)', { tracer: rec, policy: policies.teaching })
    const run = rec.run()
    const kinds = run.events.flatMap((e) => (e.type === 'message' ? [e.kind] : []))
    expect(kinds).toEqual(['call', 'exit', 'fail', 'redo', 'exit', 'call', 'exit', 'solution'])
    expect(messages(run, 'call')[0]?.payload).toEqual({ goal: 'p(1)', depth: 1, via: 'p/1 #1' })
    const schema = sldAlgorithm(consult(''), 'true').schema
    if (schema !== undefined) expect(validateAgainstSchema(run.events, schema)).toEqual([])
  })

  it('steps as an algorithm and rejects bad programs', () => {
    expect(sldAlgorithm(consult('p.'), 'p').schema?.messages['call']).toBeDefined()
    expect(() => consult('atom(x).')).toThrow(/built-in/)
    expect(() => consult('p :- .')).toThrow(/syntax/)
    expect(() => prove('loop :- loop.', 'loop', { maxSteps: 100 })).toThrow(/budget/)
  })
})
