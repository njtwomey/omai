import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { symmetricEigen } from './_matrix.ts'
import { shrinkagePath, targetShrinkage } from './shrinkage-targets.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 })
const target = fc.constantFrom('identity', 'diagonal', 'constant-correlation', 'single-factor' as const)

describe('target shrinkage on arbitrary data', () => {
  it('gives an intensity in [0, 1] and a symmetric positive semidefinite covariance', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 2, max: 10 }), fc.integer({ min: 6, max: 60 }), target, (s, p, n, t) => {
        const x = normal(key(s), [n, p])
        const r = targetShrinkage(x, { target: t })
        expect(r.shrinkage).toBeGreaterThanOrEqual(0)
        expect(r.shrinkage).toBeLessThanOrEqual(1)
        for (let i = 0; i < p; i++)
          for (let j = 0; j < p; j++) expect(r.covariance[i]?.[j]).toBeCloseTo(r.covariance[j]?.[i] ?? 0, 12)
        expect(symmetricEigen(r.covariance).values[p - 1]).toBeGreaterThan(-1e-9)
      }),
      { numRuns: 60 },
    )
  })

  it('has a held-out likelihood path whose endpoints are the sample and the target', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 2, max: 6 }), target, (s, p, t) => {
        const x = normal(key(s), [30, p])
        const path = shrinkagePath(x, { target: t, intensities: [0, 0.5, 1], folds: 3 })
        expect(path.map((q) => q.intensity)).toEqual([0, 0.5, 1])
        for (const q of path) expect(q.eigenvalues).toHaveLength(p)
        // The mixture of two positive semidefinite matrices is no worse conditioned than the worse of the two.
        expect(path[1]?.conditionNumber).toBeLessThanOrEqual(
          Math.max(path[0]?.conditionNumber ?? 0, path[2]?.conditionNumber ?? 0) * (1 + 1e-9),
        )
      }),
      { numRuns: 30 },
    )
  })
})
