import { describe, expect, it } from 'vitest'
import {
  cnfClauses,
  entails,
  evaluateFormula,
  formulaToString,
  formulaVariables,
  minimalCnf,
  minimalDnf,
  parseFormula,
  primeImplicants,
  simplifyFormula,
  toCnf,
  toDnf,
  truthTable,
} from './index.ts'

const show = (s: string): string => formulaToString(parseFormula(s))

describe('propositional logic', () => {
  it('parses with standard precedence and several notations', () => {
    expect(show('~p & q | r')).toBe('(~p & q) | r')
    expect(show('p -> q -> r')).toBe('p -> (q -> r)')
    expect(show('p <-> q <-> r')).toBe('(p <-> q) <-> r')
    expect(show('p ^ q & r')).toBe('p ^ (q & r)')
    expect(show('¬p ∧ q → r ∨ ⊥')).toBe('(~p & q) -> (r | false)')
    expect(show('not a and b or c implies d iff 1')).toBe('(((~a & b) | c) -> d) <-> true')
    expect(show('!a && b || c => d <=> 0')).toBe('(((~a & b) | c) -> d) <-> false')
    expect(() => parseFormula('p &')).toThrow(/parseFormula/)
    expect(() => parseFormula('(p')).toThrow(/expected \)/)
    expect(() => parseFormula('p $ q')).toThrow(/unexpected/)
  })

  it('evaluates and tabulates', () => {
    const f = parseFormula('(p -> q) & (q -> p)')
    expect(truthTable(f).rows.map((r) => r.value)).toEqual([true, false, false, true])
    expect(() => evaluateFormula(f, { p: true })).toThrow(/no value/)
    expect(formulaVariables(parseFormula('b | a | b'))).toEqual(['a', 'b'])
  })

  it('converts to normal forms', () => {
    expect(formulaToString(toCnf(parseFormula('p <-> q')))).toBe('(~p | q) & (p | ~q)')
    expect(formulaToString(toDnf(parseFormula('p <-> q')))).toBe('(~p & ~q) | (q & p)'.replace('(q & p)', '(p & q)'))
    expect(formulaToString(toCnf(parseFormula('p | ~p')))).toBe('true')
    expect(formulaToString(toCnf(parseFormula('p & ~p')))).toBe('p & ~p')
    expect(formulaToString(toDnf(parseFormula('p & ~p')))).toBe('false')
    expect(cnfClauses(parseFormula('p & (q | ~q)'))).toEqual([[{ name: 'p', positive: true }]])
  })

  it('simplifies and minimises', () => {
    expect(formulaToString(simplifyFormula(parseFormula('p | (p & q) | false')))).toBe('p')
    expect(formulaToString(simplifyFormula(parseFormula('(p <-> true) ^ false')))).toBe('p')
    expect(formulaToString(simplifyFormula(parseFormula('p ^ p')))).toBe('false')
    // The textbook majority function.
    const maj = parseFormula('(a & b) | (a & c) | (b & c) | (a & b & c)')
    expect(formulaToString(minimalDnf(maj))).toBe('(a & b) | (a & c) | (b & c)')
    expect(formulaToString(minimalCnf(maj))).toBe('(a | b) & (a | c) & (b | c)')
    expect(primeImplicants(parseFormula('true'), [])).toEqual([{}])
    expect(formulaToString(minimalDnf(parseFormula('p | ~p')))).toBe('true')
  })

  it('decides entailment', () => {
    const kb = ['rain -> wet', 'wet -> slippery', 'rain'].map((s) => parseFormula(s))
    expect(entails(kb, parseFormula('slippery'))).toBe(true)
    expect(entails(kb.slice(0, 2), parseFormula('slippery'))).toBe(false)
  })
})
