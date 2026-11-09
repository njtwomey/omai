import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { ardRegression } from './ard.ts'

const dataset = fc
  .tuple(fc.integer({ min: 12, max: 30 }), fc.integer({ min: 1, max: 5 }), fc.integer({ min: 0, max: 1000 }))
  .map(([n, d, seed]) => {
    const x = Array.from({ length: n }, (_, i) =>
      Array.from({ length: d }, (_, j) => Math.sin(seed + i * (j + 1.3)) + 0.1 * j),
    )
    const y = x.map((r, i) => 1.5 * (r[0] ?? 0) + 0.5 + 0.05 * Math.cos(seed * 3 + i * 2.1))
    return { x, y }
  })

describe('ard properties', () => {
  it('pruned weights are zero, the covariance is symmetric with a positive diagonal, and predictions are affine', () => {
    fc.assert(
      fc.property(dataset, (d) => {
        const f = ardRegression().fit(d)
        const { coef, keep, sigma, intercept } = f.params
        for (const [j, k] of keep.entries()) if (!k) expect(coef[j]).toBe(0)
        expect(sigma.length).toBe(keep.filter(Boolean).length)
        for (const [i, row] of sigma.entries()) {
          expect(row[i]).toBeGreaterThan(0)
          for (const [j, v] of row.entries()) expect(v).toBeCloseTo(sigma[j]?.[i] ?? 0, 8)
        }
        const pred = toFlat(f.predict(d.x))
        for (const [i, r] of d.x.entries())
          expect(pred[i]).toBeCloseTo(intercept + r.reduce((a, v, j) => a + v * (coef[j] ?? 0), 0), 8)
      }),
      { numRuns: 25 },
    )
  })

  it('permuting the features permutes the weights', () => {
    fc.assert(
      fc.property(dataset, (d) => {
        fc.pre(d.x[0]?.length === 3 || (d.x[0]?.length ?? 0) >= 2)
        const reversed = { x: d.x.map((r) => [...r].reverse()), y: d.y }
        const a = ardRegression().fit(d).params.coef
        const b = ardRegression().fit(reversed).params.coef
        for (const [j, v] of a.entries()) expect(b[a.length - 1 - j]).toBeCloseTo(v, 6)
      }),
      { numRuns: 20 },
    )
  })
})
