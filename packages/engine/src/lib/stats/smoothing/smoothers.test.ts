import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as M from './index.ts'

describe('smoothers', () => {
  it('local polynomial smoothers reproduce polynomials of their degree', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -5, max: 5 }),
        fc.integer({ min: -5, max: 5 }),
        fc.integer({ min: -3, max: 3 }),
        (a, b, c) => {
          const x = Array.from({ length: 15 }, (_, i) => i * 0.7)
          const quad = x.map((v) => a + b * v + c * v * v)
          toFlat(M.loess(x, quad, { degree: 2, span: 0.5 })).forEach((v, i) => {
            expect(v).toBeCloseTo(quad[i] ?? 0, 6)
          })
          const line = x.map((v) => a + b * v)
          toFlat(M.loess(x, line, { degree: 1, span: 0.5 })).forEach((v, i) => {
            expect(v).toBeCloseTo(line[i] ?? 0, 6)
          })
          toFlat(M.localLinear(x, line, [0.3, 5], { bandwidth: 1 })).forEach((v, i) => {
            expect(v).toBeCloseTo(a + b * ([0.3, 5][i] ?? 0), 8)
          })
        },
      ),
      { numRuns: 20 },
    )
  })

  it('robust smoothers resist outliers', () => {
    const x = Array.from({ length: 40 }, (_, i) => i / 4)
    const y = x.map((v, i) => Math.sin(v) + (i % 10 === 3 ? 8 : 0))
    const plain = toFlat(M.loess(x, y, { span: 0.3 }))
    const robust = toFlat(M.loess(x, y, { span: 0.3, family: 'symmetric' }))
    const err = (f: number[]): number => f.reduce((s, v, i) => s + Math.abs(v - Math.sin(x[i] ?? 0)), 0)
    expect(err(robust)).toBeLessThan(err(plain) / 2)
    const lw = toFlat(M.lowess(x, y, { frac: 0.3 }).fitted)
    expect(err(lw)).toBeLessThan(err(plain))
  })

  it('a smoothing spline with df degrees of freedom has that trace, and more df means less smoothing', () => {
    const x = Array.from({ length: 30 }, (_, i) => i / 3)
    const y = x.map((v, i) => Math.sin(v) + 0.3 * (toFlat(normal(key(1), [30]))[i] ?? 0))
    const s4 = M.smoothingSpline(x, y, { df: 4 })
    const s8 = M.smoothingSpline(x, y, { df: 8 })
    expect(s4.degreesOfFreedom).toBeCloseTo(4, 8)
    expect(s8.lam).toBeLessThan(s4.lam)
    expect(M.smoothingSpline(x, y).gcv).toBeGreaterThan(0)
  })

  it('running statistics leave incomplete windows empty', () => {
    const r = toFlat(M.runningMedian([1, 2, 3, 4, 5, 6], 4))
    expect(Number.isNaN(r[0] ?? 0)).toBe(true)
    expect(r[2]).toBe(2.5)
    expect(() => M.runningMean([1, 2], 0)).toThrow()
    expect(
      toFlat(M.nadarayaWatson([0, 1, 2, 3], [0, 0, 1, 1], [1.5], { kernel: 'epanechnikov', bandwidth: 2 }))[0],
    ).toBeCloseTo(0.5, 12)
  })
})
