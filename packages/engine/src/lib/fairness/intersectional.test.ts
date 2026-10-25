import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { demographicParityDifference } from './group-metrics.ts'
import {
  entropyDecomposition,
  generalisedEntropyIndex,
  intersectionalGroups,
  richSubgroupAudit,
  subgroupRates,
  theilIndex,
} from './intersectional.ts'

describe('intersectional fairness', () => {
  it('orders subgroups by each attribute, numbers numerically', () => {
    const g = intersectionalGroups([[10, 2, 10, 2]])
    expect(g.names).toEqual(['2', '10'])
    expect(() => intersectionalGroups([])).toThrow(/at least one/)
    expect(() => intersectionalGroups([[1], [1, 2]])).toThrow(/length/)
  })

  it('finds a gap hidden in an intersection that neither attribute shows alone', () => {
    // Selected when sex and age agree: each attribute alone is balanced.
    const sex = ['f', 'f', 'm', 'm', 'f', 'f', 'm', 'm']
    const age = ['o', 'y', 'o', 'y', 'o', 'y', 'o', 'y']
    const pred = [1, 0, 0, 1, 1, 0, 0, 1]
    const y = pred.map(() => 0)
    expect(demographicParityDifference(y, pred, sex).estimate.value).toBe(0)
    expect(demographicParityDifference(y, pred, age).estimate.value).toBe(0)
    const r = subgroupRates(y, pred, [sex, age])
    expect(r.gaps.selectionRate).toBe(1)
    expect(r.worst.selectionRate.gap).toBe(0.5)
    expect(subgroupRates(y, pred, [sex, age], { minSize: 3 }).gaps.selectionRate).toBe(0)
  })

  it('entropy indices vanish for equal treatment and decompose exactly (property)', () => {
    expect(generalisedEntropyIndex([1, 0], [1, 0])).toBe(0)
    expect(theilIndex([1, 0, 1], [1, 0, 1])).toBe(0)
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.boolean(), fc.integer({ min: 0, max: 2 })), { minLength: 2, maxLength: 40 }),
        fc.constantFrom(0.5, 1, 2, 3),
        (rows, alpha) => {
          const y = rows.map(([a]) => (a ? 1 : 0))
          const p = rows.map(([, b]) => (b ? 1 : 0))
          const g = rows.map(([, , c]) => c)
          const d = entropyDecomposition(y, p, g, { alpha })
          return (
            Number.isFinite(d.total) &&
            d.between >= -1e-12 &&
            Math.abs(d.total - generalisedEntropyIndex(y, p, { alpha })) < 1e-12
          )
        },
      ),
      { numRuns: 150 },
    )
  })

  it('audits false negatives as false positives with the labels flipped, and needs the audited class', () => {
    const x = [[0], [0], [0], [1]]
    const fn = richSubgroupAudit(x, [1, 1, 1, 1], [1, 1, 1, 0], { rate: 'falseNegative' })
    const fp = richSubgroupAudit(x, [0, 0, 0, 0], [0, 0, 0, 1])
    expect(fn.members).toEqual(fp.members)
    expect(fn.weightedDisparity).toBeCloseTo(fp.weightedDisparity, 15)
    expect(() => richSubgroupAudit(x, [1, 1, 1, 1], [0, 0, 0, 0])).toThrow(/audited class/)
  })
})
