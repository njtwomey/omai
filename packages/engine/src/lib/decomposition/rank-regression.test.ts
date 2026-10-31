import { describe, expect, it } from 'vitest'
import { olsSummary } from '../linear/inference.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { pcr, reducedRankRegression } from './rank-regression.ts'

const n = 80
const noise = toFlat(normal(key(3), [n * 6]))
// Features driven by two latent factors plus a little noise: strongly collinear.
const t1 = Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 2)
const t2 = Array.from({ length: n }, (_, i) => Math.cos(i * 0.91))
const x = t1.map((v, i) => [
  v + 0.05 * (noise[i * 6] ?? 0),
  0.5 * v + t2[i]! + 0.05 * (noise[i * 6 + 1] ?? 0),
  -v + 0.5 * t2[i]! + 0.05 * (noise[i * 6 + 2] ?? 0),
  t2[i]! + 0.05 * (noise[i * 6 + 3] ?? 0),
])

describe('pcr', () => {
  const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) - (r[3] ?? 0) + 0.1 * (noise[i * 6 + 4] ?? 0))

  it('is ordinary least squares with every component', () => {
    const f = pcr({ nComponents: 4 }).fit({ x, y })
    const ols = olsSummary({ x, y })
    f.params.coef.forEach((c, j) => expect(c).toBeCloseTo(ols.table.estimates[j] ?? Number.NaN, 6))
    expect(f.params.intercept).toBeCloseTo(ols.table.estimates[4] ?? Number.NaN, 6)
  })

  it('keeps orthonormal axes, ordered variances and uncorrelated scores', () => {
    const f = pcr({ nComponents: 2 }).fit({ x, y })
    const [a, b] = f.params.components
    expect((a ?? []).reduce((s, v) => s + v * v, 0)).toBeCloseTo(1, 12)
    expect((a ?? []).reduce((s, v, j) => s + v * (b?.[j] ?? 0), 0)).toBeCloseTo(0, 12)
    expect(f.params.explainedVariance[0]).toBeGreaterThan(f.params.explainedVariance[1] ?? 0)
    const scores = toFlat(f.transform(x))
    let cross = 0
    for (let i = 0; i < n; i++) cross += (scores[i * 2] ?? 0) * (scores[i * 2 + 1] ?? 0)
    expect(cross / n).toBeCloseTo(0, 10)
    expect(f.params.explainedVarianceRatio.reduce((s, v) => s + v, 0)).toBeLessThanOrEqual(1)
  })

  it('predicts from the retained components only', () => {
    const f = pcr({ nComponents: 2 }).fit({ x, y })
    const scores = toFlat(f.transform(x))
    const fitted = toFlat(f.predict(x))
    const ybar = y.reduce((s, v) => s + v, 0) / n
    for (let i = 0; i < n; i++)
      expect(fitted[i]).toBeCloseTo(ybar + (f.params.scoreCoef[0] ?? 0) * (scores[i * 2] ?? 0) + (f.params.scoreCoef[1] ?? 0) * (scores[i * 2 + 1] ?? 0), 9)
    expect(f.score({ x, y })).toBeGreaterThan(0.95)
  })

  it('rejects too many components and sample weights', () => {
    expect(() => pcr({ nComponents: 9 }).fit({ x, y })).toThrow(/nComponents/)
    expect(() => pcr().fit({ x, y, weights: y.map(() => 1) })).toThrow(/weights/)
  })

  it('in standardised units, is invariant to the scale of the features', () => {
    const scaled = x.map((r) => [r[0]! * 100, r[1]!, r[2]! * 0.01, r[3]!])
    const a = toFlat(pcr({ nComponents: 2, scale: true }).fit({ x, y }).predict(x))
    const b = toFlat(pcr({ nComponents: 2, scale: true }).fit({ x: scaled, y }).predict(scaled))
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i] ?? Number.NaN, 8))
  })
})

describe('reducedRankRegression', () => {
  // Three outputs driven by one latent direction of the features, plus noise.
  const u = [1.0, -0.5, 0.25, 2.0]
  const load = [1, -2, 0.5]
  const y = x.map((r, i) => {
    const z = r.reduce((s, v, j) => s + v * (u[j] ?? 0), 0)
    return load.map((l, c) => l * z + 0.05 * (noise[i * 6 + 4 + (c % 2)] ?? 0) + 0.02 * c)
  })

  it('is multi-output least squares at full rank', () => {
    const f = reducedRankRegression({ rank: 3 }).fit({ x, y })
    for (let c = 0; c < 3; c++) {
      const ols = olsSummary({ x, y: y.map((r) => r[c] ?? 0) })
      for (let j = 0; j < 4; j++) expect(f.params.coef[j]?.[c]).toBeCloseTo(ols.table.estimates[j] ?? Number.NaN, 6)
      expect(f.params.intercept[c]).toBeCloseTo(ols.table.estimates[4] ?? Number.NaN, 6)
    }
  })

  it('has the rank asked for and adds exactly the discarded singular values to the error (Eckart-Young)', () => {
    const full = reducedRankRegression({ rank: 3 }).fit({ x, y })
    const sse = (pred: number[]): number => y.flat().reduce((s, v, i) => s + (v - (pred[i] ?? 0)) ** 2, 0)
    const base = sse(Array.from(toFlat(full.predict(x))))
    for (const rank of [1, 2]) {
      const f = reducedRankRegression({ rank }).fit({ x, y })
      const extra = full.params.singularValues.slice(rank).reduce((s, v) => s + v * v, 0)
      expect(sse(Array.from(toFlat(f.predict(x))))).toBeCloseTo(base + extra, 6)
      // The coefficient matrix factors through `rank` directions: every (rank + 1)-minor vanishes.
      expect(f.params.inputLoadings[0]?.length).toBe(rank)
    }
    // Rank 1: all 2 x 2 minors of the coefficient matrix are zero.
    const b = reducedRankRegression({ rank: 1 }).fit({ x, y }).params.coef
    expect((b[0]?.[0] ?? 0) * (b[1]?.[1] ?? 0) - (b[0]?.[1] ?? 0) * (b[1]?.[0] ?? 0)).toBeCloseTo(0, 10)
  })

  it('recovers a planted rank-one coefficient matrix up to the noise', () => {
    const f = reducedRankRegression({ rank: 1 }).fit({ x, y })
    // The features are nearly collinear, so judge the coefficients by what they predict, not entry by entry.
    const pred = toFlat(f.predict(x))
    let err = 0
    x.forEach((r, i) => {
      const z = r.reduce((sum, v, j) => sum + v * (u[j] ?? 0), 0)
      load.forEach((l, c) => {
        err += (pred[i * 3 + c] - (l * z + 0.02 * c)) ** 2
      })
    })
    expect(Math.sqrt(err / (n * 3))).toBeLessThan(0.05)
    expect(f.score({ x, y })).toBeGreaterThan(0.99)
  })

  it('with the covariance weighting and full rank is also least squares, and at rank one differs from the identity weighting', () => {
    const cov = reducedRankRegression({ rank: 3, weighting: 'covariance' }).fit({ x, y })
    const id = reducedRankRegression({ rank: 3 }).fit({ x, y })
    cov.params.coef.forEach((row, j) => row.forEach((v, c) => expect(v).toBeCloseTo(id.params.coef[j]?.[c] ?? Number.NaN, 6)))
    // Outputs with very different scales: the weightings choose different rank-one fits.
    const skewed = y.map((r) => [r[0] ?? 0, (r[1] ?? 0) * 50, (r[2] ?? 0) + 0.3 * Math.sin((r[0] ?? 0) * 40)])
    const a = reducedRankRegression({ rank: 1 }).fit({ x, y: skewed }).params.coef
    const b = reducedRankRegression({ rank: 1, weighting: 'covariance' }).fit({ x, y: skewed }).params.coef
    expect(Math.abs((a[0]?.[2] ?? 0) - (b[0]?.[2] ?? 0))).toBeGreaterThan(1e-3)
  })

  it('shrinks the full-rank fit with a ridge penalty and validates the rank', () => {
    const ridge = reducedRankRegression({ rank: 3, alpha: 50 }).fit({ x, y }).params
    const plain = reducedRankRegression({ rank: 3 }).fit({ x, y }).params
    const norm = (c: number[][]): number => c.flat().reduce((s, v) => s + v * v, 0)
    expect(norm(ridge.coef)).toBeLessThan(norm(plain.coef))
    expect(() => reducedRankRegression({ rank: 4 }).fit({ x, y })).toThrow(/rank/)
  })
})
