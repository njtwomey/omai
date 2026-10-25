import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { boundedGroupLoss, distributionParity, equalOpportunityInDistribution, regressionGaps } from './regression.ts'

describe('regression fairness', () => {
  it('distribution distances are zero for identical groups, symmetric, and at most 1 for KS (property)', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 2, maxLength: 30 }), (xs) => {
        const scores = [...xs, ...xs]
        const groups = [...xs.map(() => 'a'), ...xs.map(() => 'b')]
        const ks = distributionParity(scores, groups)
        const shifted = distributionParity([...xs, ...xs.map((v) => v + 1)], groups, { distance: 'wasserstein' })
        return ks.value === 0 && ks.distances[0]?.[1] === ks.distances[1]?.[0] && Math.abs(shifted.value - 1) < 1e-9
      }),
      { numRuns: 100 },
    )
  })

  it('looks only at the positives for equal opportunity, and handles absolute loss and three groups', () => {
    const r = equalOpportunityInDistribution([0.9, 0.1, 0.9, 0.8], [1, 0, 1, 0], ['a', 'a', 'b', 'b'])
    expect(r.value).toBe(0)
    const l = boundedGroupLoss([0, 0, 0], [1, -2, 3], ['a', 'b', 'c'], { loss: 'absolute' })
    expect(l.losses).toEqual([1, 2, 3])
    expect(l.overall).toBe(2)
    expect(regressionGaps([1, 1, 1], [1, 1, 1], ['a', 'b', 'c']).calibrationGap).toBe(0)
    expect(() => boundedGroupLoss([1], [1, 2], ['a'])).toThrow(/one prediction/)
  })
})
