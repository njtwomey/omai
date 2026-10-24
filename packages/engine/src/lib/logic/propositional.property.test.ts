import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { formula } from '../../testing/logic.ts'
import {
  cdcl,
  countModels,
  equivalent,
  formulaToString,
  formulaVariables,
  isSatisfiable,
  minimalDnf,
  models,
  parseFormula,
  primeImplicants,
  simplifyFormula,
  toCnf,
  toDnf,
  toNnf,
  truthTable,
  tseitin,
} from './index.ts'

const NAMES = ['p', 'q', 'r', 'z']

describe('propositional properties', () => {
  it('normal forms and simplification preserve meaning; Tseitin preserves satisfiability', () => {
    fc.assert(
      fc.property(formula(NAMES, 4), (f) => {
        const table = truthTable(f, NAMES).rows.map((r) => r.value)
        for (const g of [toNnf(f), toCnf(f), toDnf(f), simplifyFormula(f), minimalDnf(f, NAMES)]) {
          expect(truthTable(g, NAMES).rows.map((r) => r.value)).toEqual(table)
        }
        const sat = table.some((x) => x)
        const t = tseitin(f)
        expect(cdcl(t.cnf).status).toBe(sat ? 'sat' : 'unsat')
        expect(isSatisfiable(f)).toBe(sat)
        const vars = formulaVariables(f)
        expect(countModels(t.cnf)).toBe(models(f, vars).length)
      }),
      { numRuns: 300 },
    )
  })

  it('prints and parses back every style but sympy', () => {
    fc.assert(
      fc.property(formula(NAMES, 4), (f) => {
        for (const style of ['ascii', 'unicode'] as const) {
          expect(equivalent(parseFormula(formulaToString(f, { style })), f)).toBe(true)
        }
        expect(formulaToString(parseFormula(formulaToString(f)))).toBe(formulaToString(f))
      }),
      { numRuns: 200 },
    )
  })

  it('prime implicants imply the formula and are prime', () => {
    fc.assert(
      fc.property(formula(['p', 'q', 'r'], 3), (f) => {
        const vars = ['p', 'q', 'r']
        for (const imp of primeImplicants(f, vars)) {
          const term = (val: Readonly<Record<string, boolean>>): boolean =>
            Object.entries(imp).every(([k, b]) => val[k] === b)
          const rows = truthTable(f, vars).rows
          expect(rows.every((r) => !term(r.valuation) || r.value)).toBe(true)
          for (const drop of Object.keys(imp)) {
            const smaller = Object.fromEntries(Object.entries(imp).filter(([k]) => k !== drop))
            const t2 = (val: Readonly<Record<string, boolean>>): boolean =>
              Object.entries(smaller).every(([k, b]) => val[k] === b)
            expect(rows.every((r) => !t2(r.valuation) || r.value)).toBe(false)
          }
        }
      }),
      { numRuns: 200 },
    )
  })
})
