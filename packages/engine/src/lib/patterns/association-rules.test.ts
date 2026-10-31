import { describe, expect, it } from 'vitest'
import { apriori, associationRules, ruleMeasures } from './index.ts'

const BASKETS = [
  [1, 3, 4],
  [2, 3, 5],
  [1, 2, 3, 5],
  [2, 5],
]

describe('rule measures', () => {
  it('match the definitions on a hand-worked table', () => {
    const m = ruleMeasures({ n: 10, both: 3, antecedent: 4, consequent: 5 })
    expect(m.support).toBeCloseTo(0.3, 12)
    expect(m.confidence).toBeCloseTo(0.75, 12)
    expect(m.lift).toBeCloseTo(1.5, 12)
    expect(m.leverage).toBeCloseTo(0.1, 12)
    expect(m.conviction).toBeCloseTo(2, 12)
    // Zhang: leverage / max(sAC (1 - sA), sA (sC - sAC)) = 0.1 / max(0.18, 0.08)
    expect(m.zhangs).toBeCloseTo(0.1 / 0.18, 12)
  })

  it('independence gives lift 1, leverage 0 and conviction 1; a perfect rule has infinite conviction', () => {
    const indep = ruleMeasures({ n: 100, both: 20, antecedent: 40, consequent: 50 })
    expect(indep.lift).toBeCloseTo(1, 12)
    expect(indep.leverage).toBeCloseTo(0, 12)
    expect(indep.conviction).toBeCloseTo(1, 12)
    expect(ruleMeasures({ n: 10, both: 4, antecedent: 4, consequent: 6 }).conviction).toBe(Number.POSITIVE_INFINITY)
  })

  it('rejects inconsistent counts', () => {
    expect(() => ruleMeasures({ n: 10, both: 5, antecedent: 4, consequent: 6 })).toThrow()
  })
})

describe('association rules', () => {
  const itemsets = apriori(BASKETS, { minSupport: 0.5 })

  it('enumerates every split of every frequent itemset of two or more items', () => {
    const all = associationRules(itemsets, 4, { minThreshold: 0 })
    // four pairs give two rules each, the triple gives six
    expect(all.length).toBe(14)
  })

  it('finds the textbook perfect rules at confidence 1, ranked', () => {
    const rules = associationRules(itemsets, 4, { minThreshold: 1 })
    const text = rules.map((r) => `${r.antecedent.join(',')}=>${r.consequent.join(',')}`).sort()
    expect(text).toEqual(['1=>3', '2,3=>5', '2=>5', '3,5=>2', '5=>2'])
  })

  it('ranks by lift when asked', () => {
    const rules = associationRules(itemsets, 4, { metric: 'lift', minThreshold: 0 })
    for (let i = 1; i < rules.length; i++) expect(rules[i - 1]?.lift).toBeGreaterThanOrEqual(rules[i]?.lift ?? 0)
    expect(rules[0]?.lift).toBeCloseTo(4 / 3, 12)
  })

  it('limits the consequent size', () => {
    const rules = associationRules(itemsets, 4, { minThreshold: 0, maxConsequent: 1 })
    expect(rules.every((r) => r.consequent.length === 1)).toBe(true)
    expect(rules.length).toBe(8 + 3)
  })

  it('needs the subsets of every itemset', () => {
    expect(() => associationRules([{ items: [1, 2], count: 2 }], 4)).toThrow()
  })
})
