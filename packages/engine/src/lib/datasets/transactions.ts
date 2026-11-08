/**
 * Market-basket transactions with planted patterns for frequent itemset and association rule mining: itemsets that
 * occur together with chosen probabilities among independent background items, with the patterns and their exact
 * expected supports as the truth.
 */

import { type CorpusData, corpus, type DatasetGenerator } from '../data/index.ts'
import { foldIn, permutation, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { stamp } from './_util.ts'

/**
 * Transactions with planted itemsets: each of `patterns` itemsets (of 2 to `maxSize` items) is added to a basket with
 * its own probability, drawn between `minSupport` and `maxSupport`, and every item is added independently with the
 * background probability. The dataset is a corpus whose documents are baskets (token ids into the item names). The truth
 * has the patterns, their planted probabilities, and `expectedSupport`, the exact probability that a basket contains an
 * itemset (summing over which patterns fire), with `supportCount` for the realised count.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { plantedTransactions } from 'omai-engine/datasets'
 * generate(plantedTransactions, { knobs: { transactions: 50, patterns: 2 } }).data.tokens?.length // => 50
 * ```
 */
export const plantedTransactions: DatasetGenerator<CorpusData> = {
  name: 'plantedTransactions',
  kind: 'corpus',
  description: 'Baskets with planted frequent itemsets among background items, with exact expected supports.',
  knobs: dict({
    transactions: int(1, 10_000_000, { label: 'transactions', description: 'The number of baskets.', default: 2000 }),
    items: int(4, 100_000, { label: 'items', description: 'The number of distinct items.', default: 40 }),
    patterns: int(0, 12, { label: 'patterns', description: 'The number of planted itemsets.', default: 4 }),
    maxSize: int(2, 12, { label: 'largest pattern', description: 'The most items in a planted itemset.', default: 4 }),
    minSupport: real(0, 1, {
      label: 'lowest pattern probability',
      description: 'Pattern probabilities are uniform between the lowest and highest.',
      default: 0.1,
    }),
    maxSupport: real(0, 1, {
      label: 'highest pattern probability',
      description: 'Pattern probabilities are uniform between the lowest and highest.',
      default: 0.4,
    }),
    background: real(0, 1, {
      label: 'background probability',
      description: 'The chance that each item is in a basket on its own.',
      default: 0.05,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['transactions'] as number
    const m = knobs['items'] as number
    const np = knobs['patterns'] as number
    const bg = knobs['background'] as number
    const lo = Math.min(knobs['minSupport'] as number, knobs['maxSupport'] as number)
    const hi = Math.max(knobs['minSupport'] as number, knobs['maxSupport'] as number)
    const maxSize = Math.min(knobs['maxSize'] as number, m)
    const patterns: number[][] = []
    const probabilities: number[] = []
    const seen = new Set<string>()
    for (let p = 0; p < np; p++) {
      const u = uniform(foldIn(foldIn(k, 0), p), [2]).data
      const size = 2 + Math.floor((u[0] ?? 0) * (maxSize - 1))
      const items = Array.from(permutation(foldIn(foldIn(k, 1), p), m).data.subarray(0, size)).sort((a, b) => a - b)
      const id = items.join(',')
      if (seen.has(id)) continue
      seen.add(id)
      patterns.push(items)
      probabilities.push(lo + (hi - lo) * (u[1] ?? 0))
    }
    const draws = uniform(foldIn(k, 2), [n, np + m]).data
    const baskets: number[][] = []
    for (let t = 0; t < n; t++) {
      const present = new Set<number>()
      patterns.forEach((items, p) => {
        if ((draws[t * (np + m) + p] ?? 1) < (probabilities[p] ?? 0)) for (const i of items) present.add(i)
      })
      for (let i = 0; i < m; i++) if ((draws[t * (np + m) + np + i] ?? 1) < bg) present.add(i)
      baskets.push([...present].sort((a, b) => a - b))
    }
    const names = Array.from({ length: m }, (_, i) => `item${String(i).padStart(String(m - 1).length, '0')}`)
    /** P(basket contains every item of `itemset`), summing over which patterns fire (patterns are independent of each other and of the background). */
    const expectedSupport = (itemset: readonly number[]): number => {
      let total = 0
      for (let mask = 0; mask < 1 << patterns.length; mask++) {
        let weight = 1
        const covered = new Set<number>()
        patterns.forEach((items, p) => {
          const fires = (mask >> p) & 1
          weight *= fires === 1 ? (probabilities[p] ?? 0) : 1 - (probabilities[p] ?? 0)
          if (fires === 1) for (const i of items) covered.add(i)
        })
        let rest = 1
        for (const i of itemset) if (!covered.has(i)) rest *= bg
        total += weight * rest
      }
      return total
    }
    const supportCount = (itemset: readonly number[]): number =>
      baskets.reduce((c, b) => c + (itemset.every((i) => b.includes(i)) ? 1 : 0), 0)
    return {
      data: corpus(
        { tokens: baskets, documents: baskets.map((b) => b.map((i) => names[i]).join(' ')), vocabulary: names },
        stamp('plantedTransactions', plantedTransactions.description, knobs),
      ),
      truth: {
        extra: { patterns, patternProbabilities: probabilities, background: bg, expectedSupport, supportCount },
      },
    }
  },
}
