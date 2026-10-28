import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const levels = (h: number, w: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(fc.integer({ min: 0, max: 255 }), { minLength: w, maxLength: w }), { minLength: h, maxLength: h })

describe('histograms', () => {
  it('equalisation and CLAHE are monotone maps onto 256 levels', () => {
    fc.assert(
      fc.property(levels(8, 9), (rows) => {
        const x = rows.map((r) => r.map((v) => v / 255))
        const flat = x.flat()
        const eq = toFlat(V.equalizeHist(x))
        for (let i = 0; i < flat.length; i++)
          for (let j = 0; j < flat.length; j++)
            if ((flat[i] ?? 0) < (flat[j] ?? 0)) expect(eq[i] ?? 0).toBeLessThanOrEqual(eq[j] ?? 0)
        for (const v of [...eq, ...toFlat(V.clahe(x, { tileGridSize: [2, 2] }))]) {
          expect(v).toBeGreaterThanOrEqual(0)
          expect(v).toBeLessThanOrEqual(1)
          expect(Math.abs(v * 255 - Math.round(v * 255))).toBeLessThan(1e-9)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('counts every pixel once and back-projects the histogram', () => {
    fc.assert(
      fc.property(levels(5, 6), fc.integer({ min: 1, max: 20 }), (rows, bins) => {
        const x = rows.map((r) => r.map((v) => v / 255))
        const h = toFlat(V.calcHist(x, { bins }))
        expect(h.reduce((a, b) => a + b, 0)).toBe(30)
        const bp = toFlat(V.calcBackProject(x, h))
        // Each pixel's back-projection is the count of its own bin, so the sum is the sum of squared counts.
        expect(bp.reduce((a, b) => a + b, 0)).toBe(h.reduce((a, b) => a + b * b, 0))
      }),
      { numRuns: RUNS },
    )
  })

  it('compares identical histograms as identical', () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 1, max: 50 }), { minLength: 4, maxLength: 12 }), (h) => {
        expect(V.compareHist(h, h, 'correlation')).toBeCloseTo(1, 12)
        expect(V.compareHist(h, h, 'chi-square')).toBe(0)
        expect(V.compareHist(h, h, 'bhattacharyya')).toBeLessThan(1e-6)
        expect(V.compareHist(h, h, 'intersection')).toBe(h.reduce((a, b) => a + b, 0))
        expect(V.compareHist(h, h, 'kl-divergence')).toBe(0)
      }),
      { numRuns: RUNS },
    )
  })

  it('matches a histogram to itself and stretches to the unit range', () => {
    const x = [
      [0.1, 0.4],
      [0.3, 0.9],
    ]
    expect(toFlat(V.matchHistograms(x, x))).toEqual(x.flat())
    const s = toFlat(V.contrastStretch(x, { low: 0, high: 100 }))
    expect(Math.min(...s)).toBe(0)
    expect(Math.max(...s)).toBe(1)
    expect(toFlat(V.normalize([1, 1], { normType: 'l1' }))).toEqual([0.5, 0.5])
    expect(() => V.calcHist(x, { ranges: [1, 0] })).toThrow()
  })
})
