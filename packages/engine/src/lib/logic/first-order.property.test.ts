import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { formula } from '../../testing/logic.ts'
import {
  clausalForm,
  type FolClause,
  type FolFormula,
  type Formula,
  formulaToString,
  isSatisfiable,
  isValid,
  parseFol,
  parseFormula,
  proveFol,
  subsumes,
} from './index.ts'

/** A propositional formula as a first-order one (propositions become 0-ary predicates). */
const toFol = (f: Formula): FolFormula => parseFol(formulaToString(f).replaceAll('^', '<->').replaceAll('xor', '<->'))

function noXor(f: Formula): Formula {
  switch (f.kind) {
    case 'var':
    case 'const':
      return f
    case 'not':
      return { kind: 'not', arg: noXor(f.arg) }
    case 'and':
    case 'or':
      return { kind: f.kind, args: f.args.map(noXor) }
    case 'xor':
      return { kind: 'not', arg: { kind: 'iff', left: noXor(f.left), right: noXor(f.right) } }
    default:
      return { kind: f.kind, left: noXor(f.left), right: noXor(f.right) }
  }
}

const clausesSat = (cs: readonly FolClause[]): boolean => {
  const text =
    cs.length === 0
      ? 'true'
      : cs
          .map((c) =>
            c.length === 0
              ? 'false'
              : `(${c.map((l) => `${l.positive ? '' : '~'}${l.atom.kind === 'atom' ? l.atom.name : ''}`).join(' | ')})`,
          )
          .join(' & ')
  return isSatisfiable(parseFormula(text))
}

describe('first-order properties on the propositional fragment', () => {
  it('clausal form preserves satisfiability', () => {
    fc.assert(
      fc.property(formula(['pa', 'qa', 'ra'], 4), (f0) => {
        const f = noXor(f0)
        expect(clausesSat(clausalForm(toFol(f)))).toBe(isSatisfiable(f))
      }),
      { numRuns: 200 },
    )
  })

  it('resolution proves exactly the valid formulas', () => {
    fc.assert(
      fc.property(formula(['pa', 'qa', 'ra'], 3), (f0) => {
        const f = noXor(f0)
        const r = proveFol([], toFol(f), { maxSteps: 5000 })
        expect(r.status).toBe(isValid(f) ? 'theorem' : 'countersatisfiable')
      }),
      { numRuns: 150 },
    )
  })

  it('a clause subsumes its instances with extra literals', () => {
    fc.assert(
      fc.property(fc.constantFrom('a', 'b', 'f(a)'), fc.constantFrom('a', 'g(b)'), (x, y) => {
        const [c] = clausalForm(parseFol('p(X, Y) | -q(Y)'))
        const [d] = clausalForm(parseFol(`p(${x}, ${y}) | -q(${y}) | r(${x})`))
        if (c === undefined || d === undefined) throw new Error('clauses')
        expect(subsumes(c, d)).not.toBeNull()
        expect(subsumes(d, c)).toBeNull()
        expect(subsumes(c, c)).not.toBeNull()
      }),
    )
  })
})
