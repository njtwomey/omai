import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { associationRules } from './association-rules.ts'
import { apriori, fpGrowth, type Itemset, marketBaskets, TEXTBOOK_BASKETS } from './itemsets.ts'

const sets = (xs: readonly Itemset[]): string[] => xs.map((s) => `${s.items.join(',')}:${s.count}`).sort()

describe('frequent itemsets and rules', () => {
  it('finds Han and Kamber’s thirteen itemsets at support 2, level by level', () => {
    const a = apriori(TEXTBOOK_BASKETS, { minSupport: 2 / 9 })
    expect(a.itemsets).toHaveLength(13)
    expect(a.levels.map((l) => [l.k, l.candidates, l.frequent])).toEqual([
      [1, 5, 5],
      [2, 10, 6],
      [3, 2, 2],
    ])
    expect(sets(a.itemsets)).toContain('I1,I2,I5:2')
    const [board] = recordsOf(a.exposition, 'boardSteps')
    expect(board?.layout.type).toBe('graph')
    expect(board?.steps.length).toBeGreaterThan(3)
    expect(sets(fpGrowth(TEXTBOOK_BASKETS, { minSupport: 2 / 9 }).itemsets)).toEqual(sets(a.itemsets))
  })

  it('agrees between Apriori and FP-growth on baskets and finds the planted rules', () => {
    const t = marketBaskets(key(3))
    const a = apriori(t, { minSupport: 0.15 })
    const f = fpGrowth(t, { minSupport: 0.15 })
    expect(sets(f.itemsets)).toEqual(sets(a.itemsets))
    expect(recordsOf(f.exposition, 'boardSteps')[0]?.steps.length).toBe(
      t.length + new Set(f.itemsets.filter((s) => s.items.length === 1).map((s) => s.items[0])).size,
    )
    const { rules, exposition } = associationRules(a.itemsets, { minConfidence: 0.6 })
    expect(rules.some((r) => r.antecedent.includes('butter') && r.consequent.includes('bread'))).toBe(true)
    for (const r of rules) expect(r.confidence).toBeGreaterThanOrEqual(0.6)
    expect(recordsOf(exposition, 'coefficientTable')[0]?.estimates[0]).toBe(rules[0]?.lift)
    const exact = associationRules(apriori(TEXTBOOK_BASKETS, { minSupport: 2 / 9 }).itemsets, { minConfidence: 1 })
    expect(exact.rules.map((r) => `${r.antecedent}=>${r.consequent}`)).toContain('I1,I5=>I2')
  })
})
