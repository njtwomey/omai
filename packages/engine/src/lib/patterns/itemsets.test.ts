import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { plantedTransactions } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  apriori,
  aprioriAlgorithm,
  closedItemsets,
  eclat,
  exhaustiveItemsets,
  fpGrowth,
  fpGrowthAlgorithm,
  maximalItemsets,
  minCount,
  supportCount,
} from './index.ts'

// The worked example of the Wikipedia article on Apriori (items 1 to 5).
const BASKETS = [
  [1, 3, 4],
  [2, 3, 5],
  [1, 2, 3, 5],
  [2, 5],
]
const text = (xs: readonly { items: readonly number[]; count: number }[]): string[] =>
  xs.map((s) => `${s.items.join(',')}:${s.count}`)

describe('frequent itemsets on a worked example', () => {
  const expected = ['1:2', '2:3', '3:3', '5:3', '1,3:2', '2,3:2', '2,5:3', '3,5:2', '2,3,5:2']

  it('all three miners find the textbook itemsets at support 0.5', () => {
    expect(text(apriori(BASKETS, { minSupport: 0.5 }))).toEqual(expected)
    expect(text(fpGrowth(BASKETS, { minSupport: 0.5 }))).toEqual(expected)
    expect(text(eclat(BASKETS, { minSupport: 0.5 }))).toEqual(expected)
    expect(text(exhaustiveItemsets(BASKETS, { minSupport: 0.5 }))).toEqual(expected)
  })

  it('closed and maximal itemsets', () => {
    const all = apriori(BASKETS, { minSupport: 0.5 })
    // {2} and {5} are absorbed by {2,5} (same count 3); {1} by {1,3}; {2,3} and {3,5} by {2,3,5}.
    expect(text(closedItemsets(all))).toEqual(['3:3', '1,3:2', '2,5:3', '2,3,5:2'])
    expect(text(maximalItemsets(all))).toEqual(['1,3:2', '2,3,5:2'])
  })

  it('the threshold rounds up with tolerance', () => {
    expect(minCount(10, 0.3)).toBe(3)
    expect(() => minCount(10, 0)).toThrow()
  })

  it('maxLength limits the itemset size in every miner', () => {
    for (const miner of [apriori, fpGrowth, eclat]) {
      expect(miner(BASKETS, { minSupport: 0.5, maxLength: 1 }).every((s) => s.items.length === 1)).toBe(true)
      expect(miner(BASKETS, { minSupport: 0.5, maxLength: 2 }).length).toBe(8)
    }
  })

  it('an empty result and empty baskets are handled', () => {
    expect(apriori([[], []], { minSupport: 0.5 })).toEqual([])
    expect(fpGrowth([[], []], { minSupport: 0.5 })).toEqual([])
    expect(eclat([[], []], { minSupport: 0.5 })).toEqual([])
  })
})

describe('the algorithms step through the mining', () => {
  it('apriori takes one step per level, with candidates and prunes counted', () => {
    const { state, steps } = runAlgorithm(aprioriAlgorithm(BASKETS, { minSupport: 0.5 }), { maxSteps: 50, seed: 0 })
    // Levels 1 to 3 find itemsets; level 4 has no candidate and ends the run.
    expect(steps).toBe(4)
    expect(state.k).toBe(4)
    expect(state.candidates).toBe(0)
    expect(state.frequent.length).toBe(9)
  })

  it('fp-growth inserts one transaction per step and then mines conditional trees', () => {
    const { state, steps } = runAlgorithm(fpGrowthAlgorithm(BASKETS, { minSupport: 0.5 }), { maxSteps: 100, seed: 0 })
    expect(state.next).toBe(4)
    expect(state.phase).toBe('done')
    expect(steps).toBeGreaterThan(5)
  })
})

describe('on planted transactions', () => {
  const { data, truth } = generate(plantedTransactions, {
    key: key(3),
    knobs: {
      transactions: 800,
      items: 20,
      patterns: 3,
      maxSize: 3,
      minSupport: 0.25,
      maxSupport: 0.45,
      background: 0.03,
    },
  })
  const patterns = ((truth.extra ?? {})['patterns'] as number[][]) ?? []
  const expectedSupport = (truth.extra ?? {})['expectedSupport'] as (s: number[]) => number

  it('the miners agree and recover every planted itemset with its expected support', () => {
    const tx = data.tokens ?? []
    const a = apriori(tx, { minSupport: 0.1 })
    expect(text(fpGrowth(tx, { minSupport: 0.1 }))).toEqual(text(a))
    expect(text(eclat(tx, { minSupport: 0.1 }))).toEqual(text(a))
    for (const p of patterns) {
      const found = a.find((s) => s.items.join() === p.join())
      expect(found).toBeDefined()
      expect((found?.count ?? 0) / tx.length).toBeCloseTo(expectedSupport(p), 1)
      expect(found?.count).toBe(supportCount(tx, p))
    }
  })
})
