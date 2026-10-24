// Test helpers for the logic domain: fast-check generators of terms, formulas and CNF, and brute-force references.
// Not published (the testing folder is excluded from the build and from barrels).
import fc from 'fast-check'
import { atom, compound, type Formula, list, num, type Term, variable } from '../lib/logic/index.ts'

/** Terms over f/1, g/2, h/3, the atoms a and b, small integers and the variables X, Y, Z (depth-limited). */
export function plainTerm(maxDepth = 3, variables: readonly string[] = ['X', 'Y', 'Z']): fc.Arbitrary<Term> {
  const leaf: fc.Arbitrary<Term> = fc.oneof(
    fc.constantFrom(...variables).map((v) => variable(v)),
    fc.constantFrom('a', 'b').map((a) => atom(a)),
    fc.integer({ min: 0, max: 3 }).map((n) => num(n)),
  )
  const tree = fc.letrec<{ term: Term }>((tie) => ({
    term: fc.oneof(
      { depthSize: 'small', withCrossShrink: true },
      leaf,
      fc
        .tuple(fc.constantFrom('f', 'g', 'h'), fc.array(tie('term'), { minLength: 1, maxLength: 3 }))
        .map(([f, args]) => compound(`${f}${args.length}`, args)),
    ),
  }))
  return tree.term.filter((t) => depth(t) <= maxDepth)
}

function depth(t: Term): number {
  if (t.kind !== 'compound') return 0
  let d = 0
  for (const a of t.args) d = Math.max(d, depth(a))
  return d + 1
}

const OPERATOR_FUNCTORS: readonly [string, number][] = [
  ['+', 2],
  ['-', 2],
  ['*', 2],
  ['^', 2],
  ['=', 2],
  [':-', 2],
  [',', 2],
  [';', 2],
  ['->', 2],
  ['is', 2],
  ['mod', 2],
  ['-', 1],
  ['\\+', 1],
  ['dynamic', 1],
  ['f', 1],
  ['g', 2],
  ['.', 2],
  ['{}', 1],
]

/** Terms with operators, lists, curly terms, negative and float numbers and atoms that need quotes. */
export function syntaxTerm(): fc.Arbitrary<Term> {
  const leaf: fc.Arbitrary<Term> = fc.oneof(
    fc.constantFrom('X', 'Y', '_G1').map((v) => variable(v)),
    fc
      .constantFrom('a', 'b', '[]', '{}', 'hello world', "it's", '+', '-', ',', '|', 'dynamic', 'mod', 'B')
      .map((a) => atom(a)),
    fc.integer({ min: -5, max: 5 }).map((n) => num(n)),
    fc.constantFrom(0.5, -2.25, 3.0, 1e10, 1e-7).map((x) => num(x, true)),
  )
  const tree = fc.letrec<{ term: Term }>((tie) => ({
    term: fc.oneof(
      { depthSize: 'small' },
      leaf,
      fc
        .tuple(fc.constantFrom(...OPERATOR_FUNCTORS), fc.array(tie('term'), { minLength: 3, maxLength: 3 }))
        .map(([[f, n], args]) => compound(f, args.slice(0, n))),
      fc.array(tie('term'), { maxLength: 3 }).map((items) => list(items)),
    ),
  }))
  return tree.term.filter((t) => depth(t) <= 4)
}

/** Propositional formulas over the given variable names (depth-limited). */
export function formula(names: readonly string[] = ['p', 'q', 'r'], maxDepth = 4): fc.Arbitrary<Formula> {
  const tree = fc.letrec<{ f: Formula }>((tie) => ({
    f: fc.oneof(
      { depthSize: 'small', maxDepth },
      fc.constantFrom(...names).map((name): Formula => ({ kind: 'var', name })),
      fc.boolean().map((value): Formula => ({ kind: 'const', value })),
      tie('f').map((a): Formula => ({ kind: 'not', arg: a })),
      fc.array(tie('f'), { minLength: 2, maxLength: 3 }).map((args): Formula => ({ kind: 'and', args })),
      fc.array(tie('f'), { minLength: 2, maxLength: 3 }).map((args): Formula => ({ kind: 'or', args })),
      fc.tuple(tie('f'), tie('f')).map(([a, b]): Formula => ({ kind: 'implies', left: a, right: b })),
      fc.tuple(tie('f'), tie('f')).map(([a, b]): Formula => ({ kind: 'iff', left: a, right: b })),
      fc.tuple(tie('f'), tie('f')).map(([a, b]): Formula => ({ kind: 'xor', left: a, right: b })),
    ),
  }))
  return tree.f
}

/** Random CNF formulas in DIMACS form: up to `maxVars` variables and `maxClauses` clauses of 1 to 3 literals. */
export function cnf(maxVars = 6, maxClauses = 18): fc.Arbitrary<{ numVars: number; clauses: number[][] }> {
  return fc.integer({ min: 1, max: maxVars }).chain((numVars) =>
    fc
      .array(
        fc.array(
          fc.tuple(fc.integer({ min: 1, max: numVars }), fc.boolean()).map(([v, s]) => (s ? v : -v)),
          { minLength: 1, maxLength: 3 },
        ),
        { maxLength: maxClauses },
      )
      .map((clauses) => ({ numVars, clauses })),
  )
}

/** Whether a CNF is satisfiable, by trying every assignment (for small variable counts). */
export function bruteForceSat(numVars: number, clauses: readonly (readonly number[])[]): boolean {
  return bruteForceCount(numVars, clauses) > 0
}

/** The number of models of a CNF over variables 1..numVars, by enumeration. */
export function bruteForceCount(numVars: number, clauses: readonly (readonly number[])[]): number {
  let count = 0
  for (let m = 0; m < 1 << numVars; m++) {
    const value = (lit: number): boolean => (((m >> (Math.abs(lit) - 1)) & 1) === 1) === lit > 0
    if (clauses.every((c) => c.some(value))) count++
  }
  return count
}
