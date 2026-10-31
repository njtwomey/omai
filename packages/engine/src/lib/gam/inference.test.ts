/**
 * Tests for inference on fitted additive models. The tests plant a known truth, simulate many datasets from it with
 * explicit keys, and check the properties the methods promise: the credible bands cover the truth about as often as
 * they say, the p-value of a smooth is about uniform under the null and small under a real effect, and the pieces
 * (the weighted chi-square tail, the Wald test of a linear term) agree with exact results.
 */

import { describe, expect, it } from 'vitest'
import { poisson } from '../dists/families/index.ts'
import { studentT } from '../dists/index.ts'
import { key, normal } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import {
  fitContext,
  fitGam,
  GAM_DEFAULTS,
  gam,
  gamProblem,
  gamSummary,
  linear,
  predictionBand,
  referenceTest,
  s,
  sampleCoefficients,
  simulateCurves,
  termBand,
  weightedChiSquareSf,
} from './index.ts'

const truth = (t: number): number => Math.sin(2 * Math.PI * t)

/** A dataset of `n` rows: x0 on a jittered grid, x1 pure noise; y = f(x0) + sd * N(0, 1). */
function simulate(seed: number, n: number, sd: number, effect = 1): { x: number[][]; y: number[] } {
  const u = normal(key(seed), [3 * n]).data as Float64Array
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n, Math.min(1, Math.max(0, 0.5 + 0.2 * (u[n + i] ?? 0)))])
  const y = x.map((r, i) => 1 + effect * truth(r[0] ?? 0) + sd * (u[2 * n + i] ?? 0))
  return { x, y }
}

describe('the weighted chi-square tail', () => {
  it('is exact for equal weights and matches a chi-square of the same mean for unequal ones', () => {
    // 3 equal weights: the chi-square with 3 degrees of freedom, P(chi2_3 > 7.814727903251179) = 0.05.
    expect(weightedChiSquareSf(7.814727903251179, [1, 1, 1])).toBeCloseTo(0.05, 9)
    // Weights (2, 1): Monte Carlo from a fixed key, 2 * z1^2 + z2^2 > 4.
    const z = normal(key(7), [400000]).data as Float64Array
    let hit = 0
    for (let i = 0; i < 200000; i++) if (2 * (z[2 * i] ?? 0) ** 2 + (z[2 * i + 1] ?? 0) ** 2 > 4) hit++
    expect(Math.abs(weightedChiSquareSf(4, [2, 1]) - hit / 200000)).toBeLessThan(0.01)
  })
})

describe('the reference test of a term', () => {
  it('is the Wald statistic on integer degrees of freedom, with a chi-square or F reference', () => {
    const X = [1, 0, 0, 1, 1, 1, 2, 1, 3, 0]
    const beta = [0.8, -0.5]
    const V = [0.1, 0.02, 0.02, 0.2]
    const known = referenceTest(X, 5, 2, beta, V, 2, 0)
    // T = beta' V^-1 beta in the coordinates of the term's columns (invariant to the QR rotation).
    const det = 0.1 * 0.2 - 0.02 ** 2
    const wald = (0.2 * 0.8 ** 2 - 2 * 0.02 * 0.8 * -0.5 + 0.1 * 0.25) / det
    expect(known.statistic).toBeCloseTo(wald, 10)
    expect(known.rank).toBe(2)
    // chi-square(2) survival is exp(-t/2).
    expect(known.pValue).toBeCloseTo(Math.exp(-wald / 2), 9)
    // With an estimated scale on many residual degrees of freedom the F reference approaches it.
    expect(referenceTest(X, 5, 2, beta, V, 2, 1e6).pValue).toBeCloseTo(known.pValue, 4)
    // The fractional rank moves continuously between the one- and two-dimensional truncations.
    const t1 = referenceTest(X, 5, 2, beta, V, 1, 0).statistic
    expect(referenceTest(X, 5, 2, beta, V, 1 + 1e-12, 0).statistic).toBeCloseTo(t1, 4)
    expect(referenceTest(X, 5, 2, beta, V, 2 - 1e-12, 0).statistic).toBeCloseTo(wald, 4)
    const half = referenceTest(X, 5, 2, beta, V, 1.5, 0)
    expect(half.rank).toBe(1.5)
    expect(half.pValue).toBeGreaterThan(0)
    expect(half.pValue).toBeLessThan(1)
  })
})

describe('the summary table', () => {
  it('has a parametric p-value equal to the classical t-test of a straight-line fit', () => {
    const n = 40
    const z = normal(key(3), [n]).data as Float64Array
    const x = Array.from({ length: n }, (_, i) => [i / (n - 1)])
    const y = x.map((r, i) => 1 + 0.7 * (r[0] ?? 0) + 0.3 * (z[i] ?? 0))
    const fitted = gam.with({ terms: [linear(0)] }).fit({ x, y })
    const row = fitted.summary().parametric[0]
    // Classical simple regression t statistic.
    const xs = x.map((r) => r[0] ?? 0)
    const xm = xs.reduce((a, b) => a + b, 0) / n
    const ym = y.reduce((a, b) => a + b, 0) / n
    const sxx = xs.reduce((a, v) => a + (v - xm) ** 2, 0)
    const slope = xs.reduce((a, v, i) => a + (v - xm) * ((y[i] ?? 0) - ym), 0) / sxx
    const rss = y.reduce((a, v, i) => a + (v - ym - slope * ((xs[i] ?? 0) - xm)) ** 2, 0)
    const se = Math.sqrt(rss / (n - 2) / sxx)
    expect(row?.statisticKind).toBe('t')
    expect(row?.estimate).toBeCloseTo(slope, 9)
    expect(row?.statistic).toBeCloseTo(slope / se, 8)
    expect(row?.pValue).toBeCloseTo(2 * item(studentT(n - 2).cdf(-Math.abs(slope / se))), 8)
  })

  it('reports small p-values for a real smooth effect, with the deviance explained', () => {
    const { x, y } = simulate(11, 120, 0.4)
    const fitted = gam.with({ terms: [s(0, { k: 8 }), s(1, { k: 6 })] }).fit({ x, y })
    const table = fitted.summary()
    expect(table.smooth.map((t) => t.label)).toEqual(['s(x0)', 's(x1)'])
    expect(table.smooth[0]?.pValue).toBeLessThan(1e-10)
    expect(table.smooth[0]?.statisticKind).toBe('F')
    expect(table.smooth[0]?.refDf).toBeGreaterThan(2)
    expect(table.devianceExplained).toBeGreaterThan(0.6)
    expect(table.edfTotal).toBeCloseTo(fitted.edf, 9)
  })
})

describe('over many simulated datasets from a known additive model', () => {
  // 150 replicates of n = 100 from y = 1 + sin(2 pi x0) + 0.4 N(0, 1), with a null covariate x1.
  const reps = 150
  const n = 100
  const grid = Array.from({ length: 30 }, (_, i) => [(i + 0.5) / 30, 0.5])
  const pValues: number[] = []
  let pointwise = 0
  let pointwiseTotal = 0
  let wholeCurve = 0
  let simultaneousCurve = 0
  for (let r = 0; r < reps; r++) {
    const data = simulate(1000 + r, n, 0.4)
    const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 8 }), s(1, { k: 5 })] }
    const problem = gamProblem(data.x, data.y, hyper)
    const ctx = fitContext(problem, fitGam(problem, hyper).state)
    // The partial effect is centred over the data: so is the truth it is compared with.
    const centre = data.x.reduce((a, row) => a + truth(row[0] ?? 0), 0) / n
    const exact = grid.map((row) => truth(row[0] ?? 0) - centre)
    const band = termBand(ctx, 0, grid)
    const wide = termBand(ctx, 0, grid, { simultaneous: { key: key(r), draws: 1500 } })
    let inside = true
    let insideWide = true
    exact.forEach((v, i) => {
      const ok = v >= (band.lower[i] ?? 0) && v <= (band.upper[i] ?? 0)
      if (ok) pointwise++
      else inside = false
      if (!(v >= (wide.lower[i] ?? 0) && v <= (wide.upper[i] ?? 0))) insideWide = false
      pointwiseTotal++
    })
    if (inside) wholeCurve++
    if (insideWide) simultaneousCurve++
    pValues.push(gamSummary(ctx).smooth[1]?.pValue ?? Number.NaN)
  }

  it('has pointwise 95% bands that cover the truth at about 95% of the points', () => {
    // Across-the-function coverage of the Bayesian band is close to nominal (Nychka 1988); the binomial error on 150
    // curves of correlated points is about 1 to 2 points, so 0.91 to 0.99 is a real test that does not flake.
    const coverage = pointwise / pointwiseTotal
    expect(coverage).toBeGreaterThan(0.91)
    expect(coverage).toBeLessThan(0.99)
  })

  it('has simultaneous bands that contain the whole curve more often than the pointwise ones', () => {
    expect(simultaneousCurve / reps).toBeGreaterThan(0.88)
    expect(simultaneousCurve).toBeGreaterThan(wholeCurve)
  })

  it('gives roughly uniform p-values to a null smooth', () => {
    const rejects = pValues.filter((p) => p < 0.05).length / reps
    // Wood's test is slightly conservative for a penalised null effect, never anti-conservative by much.
    expect(rejects).toBeLessThan(0.12)
    const median = [...pValues].sort((a, b) => a - b)[Math.floor(reps / 2)] ?? 0
    expect(median).toBeGreaterThan(0.35)
    expect(median).toBeLessThan(0.75)
  })
})

describe('posterior simulation and the prediction band', () => {
  const { x, y } = simulate(5, 80, 0.3)
  const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 7 })] }
  const problem = gamProblem(x, y, hyper)
  const ctx = fitContext(problem, fitGam(problem, hyper).state)
  const grid = Array.from({ length: 12 }, (_, i) => [(i + 0.5) / 12, 0.5])

  it('draws coefficients whose covariance is the posterior covariance', () => {
    const draws = sampleCoefficients(ctx, 20000, key(2))
    const p = problem.design.p
    for (const j of [0, 1, 3]) {
      let m = 0
      for (let d = 0; d < 20000; d++) m += draws[d * p + j] ?? 0
      m /= 20000
      let v = 0
      for (let d = 0; d < 20000; d++) v += ((draws[d * p + j] ?? 0) - m) ** 2
      v /= 20000
      expect(m).toBeCloseTo(ctx.beta[j] ?? 0, 1)
      expect(v / (ctx.vp[j * p + j] ?? 1)).toBeGreaterThan(0.95)
      expect(v / (ctx.vp[j * p + j] ?? 1)).toBeLessThan(1.05)
    }
  })

  it('has simulated curves whose spread is the standard error of the band', () => {
    const count = 8000
    const curves = simulateCurves(ctx, grid, count, key(4))
    const band = predictionBand(ctx, grid)
    const at = 5
    let m = 0
    for (let d = 0; d < count; d++) m += curves[d * grid.length + at] ?? 0
    m /= count
    let v = 0
    for (let d = 0; d < count; d++) v += ((curves[d * grid.length + at] ?? 0) - m) ** 2
    expect(Math.sqrt(v / count) / (band.se[at] ?? 1)).toBeCloseTo(1, 1)
    expect(m).toBeCloseTo(band.fit[at] ?? 0, 1)
  })

  it('maps the band to the response scale through the inverse link, in order', () => {
    const counts = x.map((r) => Math.round(Math.exp(1 + truth(r[0] ?? 0))))
    const h = { ...GAM_DEFAULTS, family: poisson(), terms: [s(0, { k: 7 })] }
    const pr = gamProblem(x, counts, h)
    const c = fitContext(pr, fitGam(pr, h).state)
    const link = predictionBand(c, grid)
    const response = predictionBand(c, grid, { scale: 'response' })
    link.lower.forEach((l, i) => {
      expect(response.lower[i]).toBeCloseTo(Math.exp(l), 10)
      expect(response.upper[i]).toBeCloseTo(Math.exp(link.upper[i] ?? 0), 10)
      expect(response.fit[i] ?? 0).toBeGreaterThan(response.lower[i] ?? 0)
    })
    expect(
      toFlat(
        gam
          .with({ family: poisson(), terms: [s(0, { k: 7 })] })
          .fit({ x, y: counts })
          .predict(grid),
      ).length,
    ).toBe(12)
  })
})
