import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { detGreedyRerank, fairMinimumTable, fairTopK, positionWeight, rankingExposure } from './ranking.ts'

describe('ranking fairness', () => {
  it('exposure is parity for an alternating ranking under no position bias, and relevance gives treatment', () => {
    const flat = rankingExposure(['a', 'b', 'a', 'b'], { bias: { kind: 'geometric', p: 1e-9 } })
    expect(flat.exposureRatio).toBeCloseTo(1, 6)
    const r = rankingExposure(['a', 'b'], { relevance: [1, 0.5] })
    expect(r.treatmentRatio).toBeCloseTo((1 / Math.log2(3) / 0.5) ** -1 * 1, 9)
    expect(positionWeight(1)).toBe(1)
  })

  it('FA*IR rankings meet the minimum table at every prefix (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.double({ min: 0, max: 1, noNaN: true }), fc.boolean()), { minLength: 10, maxLength: 40 }),
        (rows) => {
          const scores = rows.map(([s]) => s)
          const prot = rows.map(([, p]) => p)
          const k = 10
          const r = fairTopK(scores, prot, k, { p: 0.3 })
          const available = prot.filter(Boolean).length
          return (
            r.protectedCounts.every((c, i) => c >= Math.min(r.minimums[i] ?? 0, available)) && r.ranking.length === k
          )
        },
      ),
      { numRuns: 100 },
    )
    expect(fairMinimumTable(3, 0.5, 0.5)).toEqual([0, 1, 1])
    expect(() => fairMinimumTable(3, 1)).toThrow(/p is in/)
  })

  it('DetGreedy keeps every group at its floor at every prefix', () => {
    const scores = Array.from({ length: 12 }, (_, i) => 12 - i)
    const groups = scores.map((_, i) => (i < 9 ? 'a' : 'b'))
    const out = detGreedyRerank(scores, groups, { a: 0.6, b: 0.4 })
    expect(out).toHaveLength(12)
    let b = 0
    for (const [i, idx] of out.entries()) {
      if (groups[idx] === 'b') b++
      expect(b).toBeGreaterThanOrEqual(Math.min(3, Math.floor(0.4 * (i + 1))))
    }
  })
})
