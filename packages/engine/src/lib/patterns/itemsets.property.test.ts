import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { apriori, closedItemsets, eclat, exhaustiveItemsets, fpGrowth, maximalItemsets, supportCount } from './index.ts'

const baskets = fc.array(fc.array(fc.integer({ min: 0, max: 7 }), { maxLength: 6 }), { minLength: 1, maxLength: 25 })
const support = fc.constantFrom(0.1, 0.2, 0.34, 0.5, 0.75)

describe('itemset mining properties', () => {
  it('Apriori, FP-growth and Eclat equal exhaustive enumeration, counts included', () => {
    fc.assert(
      fc.property(baskets, support, (tx, minSupport) => {
        const truth = exhaustiveItemsets(tx, { minSupport })
        expect(apriori(tx, { minSupport })).toEqual(truth)
        expect(fpGrowth(tx, { minSupport })).toEqual(truth)
        expect(eclat(tx, { minSupport })).toEqual(truth)
      }),
      { numRuns: 150 },
    )
  })

  it('maxLength agrees across miners', () => {
    fc.assert(
      fc.property(baskets, support, fc.integer({ min: 1, max: 3 }), (tx, minSupport, maxLength) => {
        const truth = exhaustiveItemsets(tx, { minSupport, maxLength })
        expect(apriori(tx, { minSupport, maxLength })).toEqual(truth)
        expect(fpGrowth(tx, { minSupport, maxLength })).toEqual(truth)
        expect(eclat(tx, { minSupport, maxLength })).toEqual(truth)
      }),
      { numRuns: 100 },
    )
  })

  it('counts are true counts, closed sets determine every count, maximal sets cover every frequent set', () => {
    fc.assert(
      fc.property(baskets, support, (tx, minSupport) => {
        const all = apriori(tx, { minSupport })
        for (const s of all) expect(s.count).toBe(supportCount(tx, s.items))
        const closed = closedItemsets(all)
        for (const s of all) {
          const best = Math.max(...closed.filter((c) => s.items.every((i) => c.items.includes(i))).map((c) => c.count))
          expect(best).toBe(s.count)
        }
        const maximal = maximalItemsets(all)
        for (const s of all) expect(maximal.some((m) => s.items.every((i) => m.items.includes(i)))).toBe(true)
        expect(maximal.length).toBeLessThanOrEqual(closed.length)
      }),
      { numRuns: 100 },
    )
  })
})
