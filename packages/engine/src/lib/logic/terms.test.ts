import { describe, expect, it } from 'vitest'
import {
  atom,
  clauseToString,
  compareTerms,
  compound,
  conjunction,
  conjuncts,
  indicator,
  isGround,
  list,
  listItems,
  num,
  parseTerm,
  termEquals,
  termSize,
  termToClause,
  termToString,
  termVariables,
  variable,
} from './index.ts'

describe('terms', () => {
  it('builds and inspects terms and lists', () => {
    const t = compound('f', [atom('a'), variable('X'), list([num(1), num(2.5)], variable('T'))])
    expect(termToString(t)).toBe('f(a,X,[1,2.5|T])')
    expect(isGround(t)).toBe(false)
    expect(termVariables(t).map((v) => v.name)).toEqual(['X', 'T'])
    expect(termVariables([parseTerm('p(Y)'), parseTerm('q(X, Y)')]).map((v) => v.name)).toEqual(['Y', 'X'])
    expect(listItems(list([atom('a')]))?.length).toBe(1)
    expect(listItems(atom('a'))).toBeNull()
    expect(termSize(t)).toBe(8)
    expect(() => num(1.5, false)).toThrow(/not an integer/)
    expect(() => indicator(variable('X'))).toThrow(/not callable/)
  })

  it('orders terms in the standard order', () => {
    const order = ['_A', '-1.5', '1.0', '1', '2', "'B'", 'a', 'b', 'f(z)', 'g(a)', '[a]', 'f(a,b)']
    const terms = order.map((s) => parseTerm(s))
    const shuffled = [...terms].reverse().sort(compareTerms)
    expect(shuffled.map((x) => termToString(x))).toEqual(terms.map((x) => termToString(x)))
    expect(termEquals(parseTerm('f(X)'), parseTerm('f(X)'))).toBe(true)
  })

  it('converts between clauses, conjunctions and text', () => {
    expect(conjuncts(parseTerm('true'))).toEqual([])
    expect(termToString(conjunction([]))).toBe('true')
    expect(clauseToString(termToClause(parseTerm('p :- (q ; r), \\+ s')))).toBe('p :- q;r, \\+s.')
    expect(termToString(parseTerm("'\\n'"))).toBe("'\\n'")
    expect(termToString(parseTerm("f('a b')"), { quoted: false })).toBe('f(a b)')
  })
})
