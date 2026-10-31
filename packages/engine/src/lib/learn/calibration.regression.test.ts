import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { normal as draw, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  pitHistogram,
  pitValues,
  quantileCalibration,
  quantileCalibrationCurve,
  recalibratedQuantiles,
} from './calibration.ts'

/** Observations y_i ~ N(0, 1) and a predictive N(0, s^2) for each: overconfident for s < 1, underconfident above. */
function setting(seed: number, n: number, s: number): { y: number[]; predictive: ReturnType<typeof normal> } {
  return {
    y: toFlat(draw(key(seed), [n])),
    predictive: normal(new Array<number>(n).fill(0), new Array<number>(n).fill(s)),
  }
}

describe('pitValues and pitHistogram', () => {
  it('are uniform for the true distribution and U-shaped or hump-shaped when the spread is wrong', () => {
    const calibrated = setting(1, 4000, 1)
    const flat = pitHistogram(pitValues(calibrated.predictive, calibrated.y))
    expect(flat.deviation).toBeLessThan(0.08)
    const narrow = setting(1, 4000, 0.5)
    const u = pitHistogram(pitValues(narrow.predictive, narrow.y))
    expect(u.deviation).toBeGreaterThan(0.5)
    expect(u.density[0]).toBeGreaterThan(2)
    expect(u.density[9]).toBeGreaterThan(2)
    expect(u.density[4]).toBeLessThan(0.8)
    const wide = setting(1, 4000, 2)
    const hump = pitHistogram(pitValues(wide.predictive, wide.y))
    expect(hump.density[4]).toBeGreaterThan(1.5)
    expect(hump.density[0]).toBeLessThan(0.5)
  })

  it('counts every value once, and rejects values outside the unit interval', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 60 }),
        fc.integer({ min: 1, max: 12 }),
        (pit, bins) => {
          const h = pitHistogram(pit, bins)
          return (
            h.counts.reduce((a, b) => a + b, 0) === pit.length &&
            Math.abs(h.density.reduce((a, d) => a + d, 0) / bins - 1) < 1e-9
          )
        },
      ),
      { numRuns: 40, seed: 5 },
    )
    expect(() => pitHistogram([0.5, 1.5])).toThrow()
  })
})

describe('quantileCalibration', () => {
  it('maps nominal levels to the observed frequency and inverts it', () => {
    const c = quantileCalibration([0.1, 0.2, 0.2, 0.9])
    expect(c.knots.p).toEqual([0, 0.1, 0.2, 0.9, 1])
    expect(c.knots.r).toEqual([0, 0.25, 0.75, 1, 1])
    const expected = [0, 0.25, 0.5, 0.75, 1]
    for (const [i, v] of c.recalibrate([0, 0.1, 0.15, 0.2, 1]).entries()) expect(v).toBeCloseTo(expected[i] ?? 0, 12)
    expect(c.inverse(c.recalibrate([0.05, 0.15, 0.5]))[1]).toBeCloseTo(0.15, 12)
    expect(quantileCalibrationCurve([0.1, 0.2, 0.2, 0.9], [0.1, 0.2, 0.5, 0.95]).observed).toEqual([
      0.25, 0.75, 0.75, 1,
    ])
  })

  it('restores the coverage of the central intervals of an overconfident model on fresh data', () => {
    const fit = setting(2, 3000, 0.5)
    const calibrator = quantileCalibration(pitValues(fit.predictive, fit.y))
    const fresh = setting(3, 3000, 0.5)
    const coverage = (lower: number[], upper: number[]): number =>
      fresh.y.filter((v, i) => v >= (lower[i] ?? 0) && v <= (upper[i] ?? 0)).length / fresh.y.length
    // Raw 90% interval of the narrow model, from its own quantiles: it covers about 0.68.
    const raw = recalibratedQuantiles(fresh.predictive, quantileCalibration([0.25, 0.5, 0.75]), [0.05, 0.95])
    expect(
      coverage(
        raw.map((r) => r[0] ?? 0),
        raw.map((r) => r[1] ?? 0),
      ),
    ).toBeLessThan(0.75)
    // After recalibration, about 0.9.
    const fixed = recalibratedQuantiles(fresh.predictive, calibrator, [0.05, 0.95])
    expect(
      coverage(
        fixed.map((r) => r[0] ?? 0),
        fixed.map((r) => r[1] ?? 0),
      ),
    ).toBeGreaterThan(0.87)
    expect(
      coverage(
        fixed.map((r) => r[0] ?? 0),
        fixed.map((r) => r[1] ?? 0),
      ),
    ).toBeLessThan(0.93)
    // The recalibrated PIT, R(F(y)), is uniform on the held-out half.
    const after = calibrator.recalibrate(pitValues(fresh.predictive, fresh.y))
    expect(pitHistogram(after).deviation).toBeLessThan(0.1)
  })
})
