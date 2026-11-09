import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray } from '../tensor/index.ts'
import { nonparanormalCorrelation, nonparanormalTransform } from './rank.ts'

// Values on a grid of 0.1 in [-100, 100]: distinct values stay distinct under the monotone maps below.
const data = fc.array(
  fc.array(
    fc.integer({ min: -1000, max: 1000 }).map((v) => v / 10),
    { minLength: 3, maxLength: 3 },
  ),
  {
    minLength: 8,
    maxLength: 24,
  },
)
const maps: readonly ((v: number) => number)[] = [
  (v) => v ** 3,
  (v) => Math.exp(v / 50),
  (v) => 3 * v + 7,
  (v) => Math.atan(v / 30),
]

describe('rank-based graph input is invariant to monotone marginal transforms', () => {
  it('the Spearman, Kendall and normal-score correlations do not change, exactly', () => {
    fc.assert(
      fc.property(
        data,
        fc.integer({ min: 0, max: 3 }),
        fc.integer({ min: 0, max: 3 }),
        fc.integer({ min: 0, max: 3 }),
        (x, f, g, h) => {
          const pick = [maps[f], maps[g], maps[h]] as ((v: number) => number)[]
          const moved = x.map((r) => r.map((v, j) => (pick[j] as (v: number) => number)(v)))
          for (const method of ['spearman', 'kendall', 'normal-scores'] as const) {
            const a = nonparanormalCorrelation(x, { method, positiveDefinite: false })
            const b = nonparanormalCorrelation(moved, { method, positiveDefinite: false })
            expect(b).toEqual(a)
          }
        },
      ),
      { numRuns: 40 },
    )
  })

  it('the transform of the data is unchanged too, and each column has zero mean and unit variance', () => {
    fc.assert(
      fc.property(data, (x) => {
        const moved = x.map((r) => r.map((v, j) => (maps[j] as (v: number) => number)(v)))
        const a = toArray(nonparanormalTransform(x)) as number[][]
        expect(toArray(nonparanormalTransform(moved))).toEqual(a)
        for (let j = 0; j < 3; j++) {
          const col = a.map((r) => r[j] ?? 0)
          const mean = col.reduce((s, v) => s + v, 0) / col.length
          expect(Math.abs(mean)).toBeLessThan(1e-10)
          const sd = Math.sqrt(col.reduce((s, v) => s + (v - mean) ** 2, 0) / col.length)
          // A constant column has no spread to scale.
          if (new Set(x.map((r) => r[j])).size > 1) expect(sd).toBeCloseTo(1, 10)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('a correlation of a decreasing transform is the negative of the increasing one', () => {
    fc.assert(
      fc.property(data, (x) => {
        const flipped = x.map((r) => [-(r[0] ?? 0), r[1] ?? 0, r[2] ?? 0])
        const a = nonparanormalCorrelation(x, { method: 'spearman', positiveDefinite: false })
        const b = nonparanormalCorrelation(flipped, { method: 'spearman', positiveDefinite: false })
        for (const j of [1, 2]) expect(b[0]?.[j] ?? 0).toBeCloseTo(-(a[0]?.[j] ?? 0), 12)
      }),
      { numRuns: 30 },
    )
  })
})
