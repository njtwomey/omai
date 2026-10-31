import { describe, expect, it } from 'vitest'
import { leastSquares } from '../../testing/learn.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  conformalQuantile,
  intervalCoverage,
  splitConformalRegressor,
  weightedConformalQuantile,
  weightedConformalRegressor,
} from './conformal.ts'
import { classifierDensityRatio, ulsif } from './density-ratio.ts'

/** y = 2 x + noise whose spread grows with |x|: a shift of x towards larger |x| breaks marginal coverage. */
function sample(seed: number, n: number, mean: number): { x: number[][]; y: number[] } {
  const z = toFlat(normal(key(seed), [n]))
  const e = toFlat(normal(foldIn(key(seed), 1), [n]))
  const xs = Array.from(z, (v) => v + mean)
  return { x: xs.map((v) => [v]), y: xs.map((v, i) => 2 * v + (0.2 + 0.8 * Math.abs(v)) * (e[i] ?? 0)) }
}

const shift = 1.5

/** The true ratio of N(shift, 1) to N(0, 1) at the rows of x. */
const trueRatio = (x: unknown): Float64Array =>
  Float64Array.from((x as number[][]).map(([v = 0]) => Math.exp(shift * v - (shift * shift) / 2)))

const ones = (x: unknown): Float64Array => new Float64Array((x as number[][]).length).fill(1)

describe('weightedConformalQuantile', () => {
  it('is the split-conformal quantile with equal weights and moves with the weights', () => {
    const scores = [5, 1, 4, 2, 3, 9, 7, 8, 6]
    for (const alpha of [0.1, 0.2, 0.5]) {
      expect(
        weightedConformalQuantile(
          scores,
          scores.map(() => 1),
          1,
          alpha,
        ),
      ).toBe(conformalQuantile(scores, alpha))
    }
    // A heavy test point leaves less mass for the scores, so the quantile rises to infinity.
    expect(weightedConformalQuantile([1, 2, 3, 4], [1, 1, 1, 1], 100, 0.4)).toBe(Number.POSITIVE_INFINITY)
    // Weight on the large scores moves the quantile to them.
    expect(weightedConformalQuantile([1, 2, 3, 4], [1, 1, 1, 1], 1, 0.4)).toBe(3)
    expect(weightedConformalQuantile([1, 2, 3, 4], [1, 1, 1, 5], 1, 0.4)).toBe(4)
    expect(() => weightedConformalQuantile([1, 2], [1], 1, 0.1)).toThrow()
  })
})

describe('weightedConformalRegressor', () => {
  it('restores coverage under covariate shift that plain split conformal loses', () => {
    const alpha = 0.1
    let plain = 0
    let weighted = 0
    let estimated = 0
    const reps = 16
    for (let r = 0; r < reps; r++) {
      const train = sample(1000 + r, 200, 0)
      const calibration = sample(2000 + r, 300, 0)
      const target = sample(3000 + r, 400, shift)
      const model = leastSquares().fit(train)
      plain +=
        intervalCoverage(splitConformalRegressor(model, calibration, { alpha }).predictInterval(target.x), target.y)
          .coverage / reps
      weighted +=
        intervalCoverage(
          weightedConformalRegressor(model, calibration, { alpha, ratio: trueRatio }).predictInterval(target.x),
          target.y,
        ).coverage / reps
      // The ratio from unlabelled calibration and target covariates alone.
      const est = ulsif(calibration.x, target.x, { nCentres: 40, sigma: 0.8, lambda: 0.1 })
      estimated +=
        intervalCoverage(
          weightedConformalRegressor(model, calibration, { alpha, ratio: est.predictRatio }).predictInterval(target.x),
          target.y,
        ).coverage / reps
    }
    expect(plain).toBeLessThan(1 - alpha - 0.04)
    expect(weighted).toBeGreaterThanOrEqual(1 - alpha - 0.02)
    expect(estimated).toBeGreaterThan(plain + 0.03)
    expect(estimated).toBeGreaterThanOrEqual(1 - alpha - 0.06)
  })

  it('reduces to split conformal without shift and widens where the test point is rare under calibration', () => {
    const train = sample(1, 200, 0)
    const calibration = sample(2, 300, 0)
    const model = leastSquares().fit(train)
    const flat = weightedConformalRegressor(model, calibration, { alpha: 0.1, ratio: ones })
    const split = splitConformalRegressor(model, calibration, { alpha: 0.1 })
    expect(Array.from(flat.quantileAt([[0.3], [2]]))).toEqual([split.quantile, split.quantile])
    const toward = weightedConformalRegressor(model, calibration, { alpha: 0.1, ratio: trueRatio })
    const q = Array.from(toward.quantileAt([[-1], [3]]))
    expect(q[1]).toBeGreaterThanOrEqual(q[0] ?? 0)
    const viaClassifier = weightedConformalRegressor(model, calibration, {
      alpha: 0.1,
      ratio: classifierDensityRatio(calibration.x, sample(5, 300, shift).x).predictRatio,
    })
    expect(Number.isFinite(toFlat(viaClassifier.predictInterval([[1]]).upper)[0])).toBe(true)
  })
})
