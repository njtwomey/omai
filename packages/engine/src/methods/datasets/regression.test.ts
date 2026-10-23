import { describe, expect, it } from 'vitest'
import { type DatasetGenerator, generate, type RegressionData } from '../../lib/data/index.ts'
import { svdvals } from '../../lib/numerics/linalg/index.ts'
import { key } from '../../lib/random/index.ts'
import { sub, type Tensor, tensor, toFlat, type Value } from '../../lib/tensor/index.ts'
import {
  additiveModel,
  dopplerCurve,
  heteroscedastic,
  lowRankProfile,
  makeRegression,
  piecewiseLinear,
  quantileCurves,
  sincCurve,
  sineCurve,
  sparseUncorrelated,
  stepCurve,
} from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
const rms = (xs: readonly number[]): number => Math.sqrt(mean(xs.map((v) => v * v)))

function residuals(g: DatasetGenerator<RegressionData>, knobs: Record<string, number | string | boolean>): number[] {
  const { data, truth } = generate(g, { key: key(1), knobs })
  return toFlat(sub(data.y, truth.regressionFunction?.(data.x) as Value))
}

describe('makeRegression', () => {
  it('several targets, noise level and the true coefficients', () => {
    const { data, truth } = generate(makeRegression, {
      key: key(0),
      knobs: { n: 3000, nFeatures: 5, nInformative: 2, nTargets: 2, noise: 3, bias: 1 },
    })
    expect(data.y.shape).toEqual([3000, 2])
    const r = toFlat(sub(data.y, truth.regressionFunction?.(data.x) as Value))
    expect(Math.abs(rms(r) - 3)).toBeLessThan(4 * (3 / Math.sqrt(2 * r.length)))
    const coef = truth.extra?.['coefficients'] as number[][]
    expect(coef.slice(2).flat()).toEqual([0, 0, 0, 0, 0, 0])
  })
  it('a low-rank design follows the singular profile', () => {
    const { data } = generate(makeRegression, {
      key: key(2),
      knobs: { n: 40, nFeatures: 8, effectiveRank: 2, tailStrength: 0.2 },
    })
    toFlat(svdvals(data.x)).forEach((s, i) => {
      expect(s).toBeCloseTo(lowRankProfile(8, 2, 0.2)[i] ?? 0, 10)
    })
  })
  it('shuffled columns keep the truth aligned', () => {
    const { data, truth } = generate(makeRegression, {
      key: key(3),
      knobs: { n: 20, nFeatures: 6, nInformative: 2, shuffle: true },
    })
    const coef = truth.extra?.['coefficients'] as number[]
    const nonzero = coef.flatMap((c, j) => (c !== 0 ? [j] : []))
    expect(nonzero).toEqual(truth.relevantFeatures)
    expect(toFlat(sub(data.y, truth.regressionFunction?.(data.x) as Value)).every((v) => Math.abs(v) < 1e-9)).toBe(true)
  })
  it('sparse uncorrelated has unit noise', () => {
    expect(rms(residuals(sparseUncorrelated, { n: 4000 }))).toBeCloseTo(1, 1)
  })
})

describe('curves', () => {
  it('noise levels of the one-dimensional functions', () => {
    for (const g of [sineCurve, stepCurve, sincCurve, dopplerCurve, piecewiseLinear]) {
      const r = residuals(g, { n: 4000, noise: 0.3 })
      expect(rms(r), g.name).toBeCloseTo(0.3, 1)
      expect(Math.abs(mean(r)), g.name).toBeLessThan(4 * (0.3 / Math.sqrt(4000)))
      expect(generate(g, { knobs: { noise: 0.3 } }).truth.bayesError).toBeCloseTo(0.09, 10)
    }
  })
  it('known functions and designs', () => {
    const grid = generate(dopplerCurve, { knobs: { n: 5, design: 'grid' } }).data.x
    expect(toFlat(grid)).toEqual([0, 0.25, 0.5, 0.75, 1])
    const step = generate(stepCurve, { knobs: { steps: 3, noise: 0 } }).truth.regressionFunction
    expect(toFlat(step?.(tensor([[0.1], [0.3], [0.6], [0.99]])) as Value)).toEqual([0, 1 / 3, 2 / 3, 1])
    const pw = generate(piecewiseLinear, { key: key(4), knobs: { knots: 2 } }).truth
    const knots = pw.extra?.['knots'] as number[]
    const f = (x: number) => toFlat(pw.regressionFunction?.(tensor([[x]])) as Value)[0] ?? 0
    const at = knots[0] ?? 0.5
    expect(Math.abs(f(at + 1e-9) - f(at - 1e-9))).toBeLessThan(1e-7)
  })
  it('heteroscedastic noise grows with x', () => {
    const { data, truth } = generate(heteroscedastic, { key: key(5), knobs: { n: 6000, base: 0.1, slope: 1 } })
    const x = toFlat(data.x)
    const r = toFlat(sub(data.y, truth.regressionFunction?.(data.x) as Value))
    expect(rms(r.filter((_, i) => (x[i] ?? 0) < 0.1))).toBeCloseTo(0.15, 1)
    expect(rms(r.filter((_, i) => (x[i] ?? 0) > 0.9))).toBeCloseTo(1.05, 1)
  })
  it('conditional quantiles are calibrated for every noise', () => {
    for (const noise of ['gaussian', 'laplace', 'student-t', 'gumbel']) {
      const { data, truth } = generate(quantileCurves, { key: key(6), knobs: { n: 6000, noise } })
      for (const q of [0.1, 0.5, 0.9]) {
        const bound = toFlat(truth.quantile?.(data.x, q) as Value)
        const below = mean(toFlat(data.y).map((v, i) => (v <= (bound[i] ?? 0) ? 1 : 0)))
        expect(Math.abs(below - q), `${noise} ${q}`).toBeLessThan(4 * Math.sqrt((q * (1 - q)) / 6000))
      }
      // The joint density integrates to one in y at a fixed x (trapezoid rule).
      const ys = Array.from({ length: 4001 }, (_, i) => -20 + (40 * i) / 4000)
      const xs = tensor(ys.map(() => [3]))
      const dens = toFlat(truth.logProb?.(xs, tensor(ys)) as Value).map(Math.exp)
      let area = 0
      for (let i = 1; i < ys.length; i++) area += 0.5 * ((dens[i] ?? 0) + (dens[i - 1] ?? 0)) * 0.01
      expect(area * 10, noise).toBeCloseTo(1, 3)
    }
  })
  it('additive components sum to the regression function', () => {
    const { data, truth } = generate(additiveModel, { key: key(7), knobs: { n: 50, nFeatures: 6, nInformative: 5 } })
    const parts = truth.extra?.['components'] as { f: (x: Value) => Value }[]
    const total = parts.map((p) => toFlat(p.f(data.x))).reduce((a, b) => a.map((v, i) => v + (b[i] ?? 0)))
    const f = toFlat(truth.regressionFunction?.(data.x) as Value)
    total.forEach((v, i) => {
      expect(v).toBeCloseTo(f[i] ?? 0, 12)
    })
    // Each component is centred on [0, 1].
    const grid = tensor(Array.from({ length: 2001 }, (_, i) => new Array(6).fill(i / 2000)))
    for (const p of parts) expect(Math.abs(mean(toFlat(p.f(grid))))).toBeLessThan(5e-3)
    expect(truth.relevantFeatures).toEqual([0, 1, 2, 3, 4])
    void (data.x as Tensor)
  })
})
