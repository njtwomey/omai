import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { designOf } from './design.ts'
import {
  breuschPagan,
  conditionIndices,
  hatMatrix,
  leverage,
  olsInfluence,
  regressionDiagnostics,
  varianceInflationFactors,
  whiteTest,
} from './diagnostics.ts'
import { leastSquaresSolve } from './solvers.ts'

const x = [
  [1, 0.5],
  [2, -0.3],
  [3, 0.8],
  [4, 0.1],
  [5, -0.7],
  [6, 0.4],
  [7, 1.1],
]
const y = [2.1, 3.9, 6.3, 7.8, 10.4, 17.5, 14.2] // observation 5 is an outlier

const fit = (rows: number[]): { coef: number[]; intercept: number } => {
  const r = leastSquaresSolve(
    designOf({ x: rows.map((i) => x[i] ?? []), y: rows.map((i) => y[i] ?? 0) }, 'test'),
    0,
    true,
    'svd',
  )
  return { coef: Array.from(r.coef), intercept: r.intercept }
}

describe('influence measures', () => {
  it('has a symmetric idempotent hat matrix whose trace is the rank', () => {
    const h = toFlat(hatMatrix(x))
    const n = x.length
    let trace = 0
    for (let i = 0; i < n; i++) {
      trace += h[i * n + i] ?? 0
      for (let j = 0; j < n; j++) {
        expect(h[i * n + j]).toBeCloseTo(h[j * n + i] ?? 0, 12)
        let sq = 0
        for (let k = 0; k < n; k++) sq += (h[i * n + k] ?? 0) * (h[k * n + j] ?? 0)
        expect(sq).toBeCloseTo(h[i * n + j] ?? 0, 10)
      }
    }
    expect(trace).toBeCloseTo(3, 10)
    expect(leverage(x).reduce((a, b) => a + b, 0)).toBeCloseTo(3, 10)
  })

  it('agrees with refitting the model without each observation', () => {
    const infl = olsInfluence({ x, y })
    const full = fit([0, 1, 2, 3, 4, 5, 6])
    const p = 3
    const dfbeta = toFlat(infl.dfbeta)
    for (let i = 0; i < x.length; i++) {
      const loo = fit([0, 1, 2, 3, 4, 5, 6].filter((k) => k !== i))
      const fullVec = [...full.coef, full.intercept]
      const looVec = [...loo.coef, loo.intercept]
      for (let j = 0; j < p; j++) expect(dfbeta[i * p + j]).toBeCloseTo((fullVec[j] ?? 0) - (looVec[j] ?? 0), 9)
      // The leave-one-out prediction of the held-out point gives DFFITS = (yhat - yhat_(i)) / (sigma_(i) sqrt(h)).
      const row = x[i] ?? []
      const predFull = full.intercept + (full.coef[0] ?? 0) * (row[0] ?? 0) + (full.coef[1] ?? 0) * (row[1] ?? 0)
      const predLoo = loo.intercept + (loo.coef[0] ?? 0) * (row[0] ?? 0) + (loo.coef[1] ?? 0) * (row[1] ?? 0)
      const h = infl.leverage[i] ?? 0
      expect((predFull - predLoo) / ((infl.sigmaLoo[i] ?? 1) * Math.sqrt(h))).toBeCloseTo(infl.dffits[i] ?? 0, 8)
    }
    // The outlier has the largest studentised residual (the end point has more leverage, so more Cook's distance).
    const absExternal = infl.external.map(Math.abs)
    expect(absExternal.indexOf(Math.max(...absExternal))).toBe(5)
    expect(Math.abs(infl.external[5] ?? 0)).toBeGreaterThan(Math.abs(infl.internal[5] ?? 0))
    expect(infl.dfResid).toBe(4)
  })

  it('relates the internal and external residuals, DFFITS and Cook by their closed forms', () => {
    const infl = olsInfluence({ x, y })
    for (let i = 0; i < x.length; i++) {
      const r = infl.internal[i] ?? 0
      const t = infl.external[i] ?? 0
      // t = r sqrt((n - p - 1) / (n - p - r^2))
      expect(t).toBeCloseTo(r * Math.sqrt((infl.dfResid - 1) / (infl.dfResid - r * r)), 9)
      const h = infl.leverage[i] ?? 0
      expect(infl.cooksDistance[i]).toBeCloseTo(((r * r) / infl.rank) * (h / (1 - h)), 12)
      expect(infl.covRatio[i]).toBeCloseTo(((infl.dfResid - 1 + t * t) / infl.dfResid) ** -infl.rank / (1 - h), 8)
    }
  })

  it('uses the pseudo-inverse for a rank-deficient design and whitens weights', () => {
    const dup = x.map(([a = 0]) => [a, 2 * a])
    const infl = olsInfluence({ x: dup, y })
    expect(infl.rank).toBe(2)
    expect(infl.leverage.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 10)
    // Duplicating a row is the same as giving it weight two.
    const w = x.map((_, i) => (i === 2 ? 2 : 1))
    const weighted = olsInfluence({ x, y, weights: w })
    const rep = [...x, x[2] ?? []]
    const repeated = olsInfluence({ x: rep, y: [...y, y[2] ?? 0] })
    expect(weighted.sigma).toBeCloseTo(repeated.sigma * Math.sqrt(repeated.dfResid / weighted.dfResid), 10)
    expect(weighted.leverage[2]).toBeCloseTo((repeated.leverage[2] ?? 0) + (repeated.leverage[7] ?? 0), 10)
  })
})

describe('collinearity and heteroscedasticity', () => {
  it('finds the variance inflation factor from the other columns', () => {
    const z = [
      [1, 2.1, 5],
      [2, 3.9, 1],
      [3, 6.2, 4],
      [4, 7.8, 2],
      [5, 10.1, 6],
      [6, 11.8, 3],
    ]
    const v = varianceInflationFactors(z)
    // Regress column 0 on columns 1 and 2 by least squares and use 1 / (1 - R^2).
    const col = (j: number): number[] => z.map((r) => r[j] ?? 0)
    const f = leastSquaresSolve(
      designOf({ x: z.map((r) => [r[1] ?? 0, r[2] ?? 0]), y: col(0) }, 'test'),
      0,
      true,
      'svd',
    )
    const mean = col(0).reduce((a, b) => a + b, 0) / z.length
    let rss = 0
    let tss = 0
    z.forEach((r, i) => {
      rss += ((r[0] ?? 0) - f.intercept - (f.coef[0] ?? 0) * (r[1] ?? 0) - (f.coef[1] ?? 0) * (r[2] ?? 0)) ** 2
      tss += ((r[0] ?? 0) - mean) ** 2
      void i
    })
    expect(v[0]).toBeCloseTo(1 / (rss / tss), 8)
    expect(Math.min(...v)).toBeGreaterThanOrEqual(1)
    expect(
      varianceInflationFactors([
        [1, 2],
        [2, 4],
        [3, 6],
      ])[0],
    ).toBe(Number.POSITIVE_INFINITY)
  })

  it('has condition indices that flag a near dependency, with proportions summing to one', () => {
    const ci = conditionIndices([
      [1, 2],
      [2, 4.001],
      [3, 6],
      [4, 8.001],
      [5, 9.999],
    ])
    expect(ci.conditionNumber).toBeGreaterThan(50)
    const p = toFlat(ci.varianceProportions)
    for (let j = 0; j < 3; j++) expect((p[j] ?? 0) + (p[3 + j] ?? 0) + (p[6 + j] ?? 0)).toBeCloseTo(1, 12)
  })

  it('rejects constant variance when the noise grows with the regressor', () => {
    const xs = Array.from({ length: 40 }, (_, i) => [(i + 1) / 4])
    const ys = xs.map(([v = 0], i) => 1 + 2 * v + v * v * 0.3 * (i % 2 === 0 ? 1 : -1))
    const bp = breuschPagan({ x: xs, y: ys })
    const w = whiteTest({ x: xs, y: ys })
    expect(bp.pValue).toBeLessThan(0.01)
    expect(w.pValue).toBeLessThan(0.01)
    expect(w.df).toBe(2)
    const flat = ys.map((_, i) => 1 + 2 * (xs[i]?.[0] ?? 0) + (i % 2 === 0 ? 0.1 : -0.1))
    expect(breuschPagan({ x: xs, y: flat }).pValue).toBeGreaterThan(0.05)
  })

  it('bundles a report that flags the outlier', () => {
    const r = regressionDiagnostics({ x, y })
    expect(r.influentialPoints).toContain(5)
    expect(r.vif).toHaveLength(2)
    expect(r.durbinWatson.statistic).toBeGreaterThan(0)
  })
})
