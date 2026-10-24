import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { huber, ransac, trimMean, tukeyBiweight, winsorize, winsorizedMean } from './robust.ts'

type Point = readonly [number, number]
const line = {
  fit: (p: readonly Point[]): Point => {
    const n = p.length
    const mx = p.reduce((s, q) => s + q[0], 0) / n
    const my = p.reduce((s, q) => s + q[1], 0) / n
    let sxy = 0
    let sxx = 0
    for (const [u, v] of p) {
      sxy += (u - mx) * (v - my)
      sxx += (u - mx) ** 2
    }
    const slope = sxx === 0 ? 0 : sxy / sxx
    return [my - slope * mx, slope]
  },
  residual: (m: Point, p: Point): number => p[1] - m[0] - m[1] * p[0],
}

describe('robust estimators', () => {
  it('are location and scale equivariant', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -50, max: 50, noNaN: true }), { minLength: 8, maxLength: 30 }),
        fc.double({ min: -10, max: 10, noNaN: true }),
        fc.double({ min: 0.5, max: 5, noNaN: true }),
        (x, a, b) => {
          const y = x.map((v) => a + b * v)
          expect(trimMean(y, 0.1)).toBeCloseTo(a + b * trimMean(x, 0.1), 6)
          expect(winsorizedMean(y, [0.1, 0.1])).toBeCloseTo(a + b * winsorizedMean(x, [0.1, 0.1]), 6)
        },
      ),
      { numRuns: 40 },
    )
    const x = toFlat(normal(key(30), [25]))
    expect(huber(x.map((v) => 3 + 2 * v)).value).toBeCloseTo(3 + 2 * huber(x).value, 6)
  })

  it('resist gross outliers, and winsorized values stay inside the bulk', () => {
    const x = toFlat(normal(key(31), [50]))
    const dirty = [...x.slice(0, 45), 1e3, 2e3, -1e3, 5e3, 1e4]
    expect(Math.abs(tukeyBiweight(dirty).value)).toBeLessThan(0.6)
    expect(Math.abs(huber(dirty).value)).toBeLessThan(0.8)
    const top = toFlat(winsorize(dirty, [0.1, 0.1])).reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY)
    expect(top).toBeLessThan(1e3)
  })

  it('RANSAC finds the inlier line among outliers, reproducibly per key', () => {
    const pts: Point[] = Array.from({ length: 40 }, (_, i): Point => [i, 2 * i + 1])
    for (let i = 0; i < 12; i++) pts.push([i * 3, 100 - 7 * i])
    const options = { minSamples: 2, residualThreshold: 0.5, maxTrials: 200, key: foldIn(key(5), 1) }
    const r = ransac(pts, line, options)
    expect(r.model[1]).toBeCloseTo(2, 10)
    expect(r.inliers.length).toBe(40)
    expect(ransac(pts, line, options).trials).toBe(r.trials)
  })
})
