import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  countModels,
  entails,
  equivalent,
  type Formula,
  findModel,
  isSatisfiable,
  isValid,
  minimalCnf,
  minimalDnf,
  models,
  parseFormula,
  simplifyFormula,
  toCnf,
  toDnf,
  tseitin,
} from './index.ts'

/** (terms, literals) of a sum-of-products or product-of-sums formula. */
function counts(f: Formula): [number, number] {
  if (f.kind === 'const') return [0, 0]
  const terms = f.kind === 'and' || f.kind === 'or' ? f.args : [f]
  const lits = terms.reduce((s, t) => s + (t.kind === 'and' || t.kind === 'or' ? t.args.length : 1), 0)
  return [terms.length, lits]
}

const isLiteral = (f: Formula): boolean => f.kind === 'var' || (f.kind === 'not' && f.arg.kind === 'var')
const isClauseOf = (f: Formula, kind: 'and' | 'or'): boolean =>
  isLiteral(f) || f.kind === 'const' || (f.kind === kind && f.args.every(isLiteral))

describe('propositional logic against sympy.logic', () => {
  for (const c of loadFixture('logic/propositional').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'entails') {
        const premises = (p['premises'] as string[]).map((s) => parseFormula(s))
        expect(entails(premises, parseFormula(p['conclusion'] as string))).toBe(p['entailed'])
        return
      }
      const f = parseFormula(p['formula'] as string)
      const variables = p['variables'] as string[]
      expect(models(f, variables)).toEqual(p['models'])
      expect(isSatisfiable(f)).toBe(p['satisfiable'])
      expect(isValid(f)).toBe(p['valid'])
      const m = findModel(f)
      expect(m === null).toBe(!p['satisfiable'])
      for (const g of [toCnf(f), toDnf(f), simplifyFormula(f), minimalDnf(f, variables), minimalCnf(f, variables)]) {
        expect(models(g, variables)).toEqual(p['models'])
      }
      const cnf = toCnf(f)
      expect(cnf.kind !== 'and' ? isClauseOf(cnf, 'or') : cnf.args.every((a) => isClauseOf(a, 'or'))).toBe(true)
      const dnf = toDnf(f)
      expect(dnf.kind !== 'or' ? isClauseOf(dnf, 'and') : dnf.args.every((a) => isClauseOf(a, 'and'))).toBe(true)
      // Tseitin: one extension per model when every original variable occurs.
      expect(countModels(tseitin(f).cnf)).toBe((p['models'] as unknown[]).length)
      // Exact minimisation is never larger than sympy's simplify_logic.
      const [sopTerms, sopLits] = p['minimalDnf'] as [number, number]
      const [ours, oursLits] = counts(minimalDnf(f, variables))
      expect(ours < sopTerms || (ours === sopTerms && oursLits <= sopLits)).toBe(true)
      const [posTerms, posLits] = p['minimalCnf'] as [number, number]
      const [oc, ocLits] = counts(minimalCnf(f, variables))
      expect(oc < posTerms || (oc === posTerms && ocLits <= posLits)).toBe(true)
      expect(equivalent(f, minimalDnf(f, variables))).toBe(true)
    })
  }
})
