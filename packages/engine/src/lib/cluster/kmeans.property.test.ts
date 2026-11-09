import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { kMeans, kmeansPlusPlus, nearestCentres } from './kmeans.ts'

const points = fc.tuple(fc.integer({ min: 4, max: 25 }), fc.integer({ min: 1, max: 3 })).chain(([n, d]) =>
  fc.array(fc.array(fc.double({ min: -10, max: 10, noNaN: true }), { minLength: d, maxLength: d }), {
    minLength: n,
    maxLength: n,
  }),
)

describe('k-means properties', () => {
  it('k-means++ returns k rows of the data, by index', () => {
    fc.assert(
      fc.property(points, fc.integer({ min: 1, max: 4 }), fc.integer({ min: 0, max: 500 }), (x, k, seed) => {
        const { centres, indices } = kmeansPlusPlus(x, k, { key: key(seed) })
        expect(centres.length).toBe(k)
        for (const [j, i] of indices.entries()) expect(centres[j]).toEqual(x[i])
      }),
      { numRuns: 40 },
    )
  })

  it('a fitted k-means labels every row by its nearest centre, and its inertia is their squared distance', () => {
    fc.assert(
      fc.property(points, fc.integer({ min: 1, max: 4 }), fc.integer({ min: 0, max: 500 }), (x, k, seed) => {
        const f = kMeans.with({ nClusters: k }).fit({ x }, { key: key(seed) })
        const near = nearestCentres(x, f.params.centres)
        expect(f.params.labels).toEqual(near.labels)
        expect(f.params.inertia).toBeCloseTo(near.inertia, 8)
        const dist = toArray(f.transform(x)) as number[][]
        const direct = dist.reduce((a, r, i) => a + (r[f.params.labels[i] ?? 0] ?? 0) ** 2, 0)
        expect(direct).toBeCloseTo(f.params.inertia, 6)
        expect(f.score({ x })).toBeCloseTo(-f.params.inertia, 8)
      }),
      { numRuns: 40 },
    )
  })
})
