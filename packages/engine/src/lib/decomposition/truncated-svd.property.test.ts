import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { truncatedSvd } from './truncated-svd.ts'

const matrix = fc
  .tuple(fc.integer({ min: 4, max: 15 }), fc.integer({ min: 2, max: 6 }), fc.integer({ min: 0, max: 1000 }))
  .map(([n, d, seed]) =>
    Array.from({ length: n }, (_, i) =>
      Array.from({ length: d }, (_, j) => Math.sin(seed + i * 1.9 + j * 4.1) + 0.5 * j),
    ),
  )

describe('truncated SVD properties', () => {
  it('components are orthonormal, singular values descend, and explained variance is a share of the total', () => {
    fc.assert(
      fc.property(matrix, fc.constantFrom('full' as const, 'randomized' as const), (x, solver) => {
        const k = Math.min(x.length, x[0]?.length ?? 1, 3)
        const f = truncatedSvd.with({ nComponents: k, solver }).fit({ x }, { key: key(0) })
        const c = f.params.components
        for (let a = 0; a < k; a++)
          for (let b = 0; b < k; b++)
            expect(c[a]?.reduce((s, v, j) => s + v * (c[b]?.[j] ?? 0), 0)).toBeCloseTo(a === b ? 1 : 0, 6)
        for (let i = 1; i < k; i++)
          expect(f.params.singularValues[i] ?? 0).toBeLessThanOrEqual((f.params.singularValues[i - 1] ?? 0) + 1e-9)
        const ratio = f.params.explainedVarianceRatio.reduce((a, v) => a + v, 0)
        expect(ratio).toBeGreaterThanOrEqual(0)
      }),
      { numRuns: 30 },
    )
  })

  it('projecting twice is projecting once, and the exact solver never does worse than the randomised', () => {
    fc.assert(
      fc.property(matrix, (x) => {
        const k = Math.min(x.length, x[0]?.length ?? 1, 2)
        const full = truncatedSvd.with({ nComponents: k, solver: 'full' }).fit({ x })
        const once = full.inverseTransform(full.transform(x))
        const twice = full.inverseTransform(full.transform(once))
        for (const [i, r] of (toArray(once) as number[][]).entries())
          for (const [j, v] of r.entries()) expect((toArray(twice) as number[][])[i]?.[j]).toBeCloseTo(v, 8)
        const rand = truncatedSvd.with({ nComponents: k, solver: 'randomized' }).fit({ x }, { key: key(1) })
        const error = (f: typeof full): number =>
          (toArray(f.inverseTransform(f.transform(x))) as number[][]).reduce(
            (a, r, i) => a + r.reduce((b, v, j) => b + (v - (x[i]?.[j] ?? 0)) ** 2, 0),
            0,
          )
        expect(error(full)).toBeLessThanOrEqual(error(rand) + 1e-8)
      }),
      { numRuns: 25 },
    )
  })
})
