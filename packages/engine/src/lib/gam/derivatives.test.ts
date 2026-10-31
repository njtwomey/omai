/**
 * Tests for the derivative and difference of smooths and for the partial effects with standard errors: the derivative
 * against the derivative of the planted truth and its band's coverage over many simulated datasets, Richardson
 * extrapolation against plain differences, the difference of two by-smooths against the planted difference (including
 * the null of equal curves), and the term predictions with terms left out.
 */

import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import {
  factor,
  fitContext,
  fitGam,
  GAM_DEFAULTS,
  gam,
  gamProblem,
  linear,
  predictionBand,
  predictTermsSe,
  s,
  smoothDerivative,
  smoothDifference,
} from './index.ts'

const noise = (seed: number, n: number): Float64Array => normal(key(seed), [n]).data as Float64Array
const ctxOf = (x: number[][], y: number[], terms: ReturnType<typeof s>[]) => {
  const hyper = { ...GAM_DEFAULTS, terms }
  const problem = gamProblem(x, y, hyper)
  return fitContext(problem, fitGam(problem, hyper).state)
}

describe('the derivative of a smooth', () => {
  const n = 150
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
  const grid = Array.from({ length: 25 }, (_, i) => 0.05 + (0.9 * i) / 24)
  const slope = (t: number): number => 2 * Math.PI * Math.cos(2 * Math.PI * t)

  it('has a 95% band that covers the slope of the truth at about 95% of points over 60 datasets', () => {
    let hit = 0
    let total = 0
    for (let r = 0; r < 60; r++) {
      const e = noise(900 + r, n)
      const ctx = ctxOf(
        x,
        x.map((row, i) => Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.2 * (e[i] ?? 0)),
        [s(0, { k: 10 })],
      )
      const band = smoothDerivative(ctx, 0, grid)
      grid.forEach((g, i) => {
        total++
        if (slope(g) >= (band.lower[i] ?? 0) && slope(g) <= (band.upper[i] ?? 0)) hit++
      })
    }
    // The band is for the derivative of the penalised fit, which is biased towards smoothness where the curve bends
    // fastest, so coverage is a little below nominal; 0.85 to 0.99 is the range that is a real check.
    expect(hit / total).toBeGreaterThan(0.85)
    expect(hit / total).toBeLessThan(0.99)
  })

  it('differentiates the basis to fourth order by Richardson extrapolation, second order by central differences', () => {
    const e = noise(31, n)
    const ctx = ctxOf(
      x,
      x.map((row, i) => Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.05 * (e[i] ?? 0)),
      [s(0, { k: 10 })],
    )
    const reference = smoothDerivative(ctx, 0, grid, { method: 'central', step: 1e-4 })
    const coarseCentral = smoothDerivative(ctx, 0, grid, { method: 'central', step: 0.05 })
    const coarseRichardson = smoothDerivative(ctx, 0, grid, { method: 'richardson', step: 0.05 })
    const err = (b: typeof reference): number =>
      Math.max(...Array.from(b.fit, (v, i) => Math.abs(v - (reference.fit[i] ?? 0))))
    expect(err(coarseRichardson)).toBeLessThan(err(coarseCentral))
    expect(err(coarseRichardson)).toBeLessThan(0.02)
  })

  it('gives the curvature of a quadratic as a constant, and refuses a term that is not a smooth of one covariate', () => {
    const ctx = ctxOf(
      x,
      x.map((row) => 3 * (row[0] ?? 0) ** 2),
      [s(0, { k: 8 })],
    )
    const c = smoothDerivative(ctx, 0, [0.3, 0.5, 0.7], { order: 2 })
    for (const v of c.fit) expect(v).toBeCloseTo(6, 0)
    const lin = ctxOf(
      x,
      x.map((row) => row[0] ?? 0),
      [linear(0)],
    )
    expect(() => smoothDerivative(lin, 0, [0.5])).toThrow(/one covariate/)
  })
})

describe('the difference between two smooths', () => {
  const n = 160
  const x = Array.from({ length: n }, (_, i) => [(i % 80) / 79, i < 80 ? 1 : 0, i < 80 ? 0 : 1])
  const grid = Array.from({ length: 20 }, (_, i) => 0.05 + (0.9 * i) / 19)
  const terms = (): ReturnType<typeof s>[] => [
    s(0, { k: 8, by: 1, label: 'a' }),
    s(0, { k: 8, by: 2, label: 'b' }),
    linear(1),
  ]

  it('covers the planted difference and the null difference with the simultaneous band', { timeout: 120000 }, () => {
    let coversTruth = 0
    let coversNull = 0
    let simultaneousNull = 0
    const reps = 30
    for (let r = 0; r < reps; r++) {
      const e = noise(1200 + r, n)
      const curve = (t: number, group: number): number =>
        group === 1 ? Math.sin(2 * Math.PI * t) : 0.5 * Math.sin(2 * Math.PI * t) + 0.5 * t
      const y = x.map((row, i) => curve(row[0] ?? 0, row[1] === 1 ? 1 : 2) + 0.2 * (e[i] ?? 0))
      const ctx = ctxOf(x, y, terms())
      const band = smoothDifference(ctx, 0, 1, grid)
      let ok = true
      grid.forEach((g, i) => {
        const truth = curve(g, 1) - curve(g, 2)
        if (!(truth >= (band.lower[i] ?? 0) && truth <= (band.upper[i] ?? 0))) ok = false
      })
      if (ok) coversTruth++
      // Equal curves: the difference is zero everywhere.
      const same = x.map((row, i) => Math.sin(2 * Math.PI * (row[0] ?? 0)) + 0.2 * (e[i] ?? 0))
      const nullCtx = ctxOf(x, same, terms())
      const wide = smoothDifference(nullCtx, 0, 1, grid, { simultaneous: { key: key(r), draws: 1000 } })
      if (wide.lower.every((l, i) => l <= 0 && (wide.upper[i] ?? 0) >= 0)) simultaneousNull++
      const narrow = smoothDifference(nullCtx, 0, 1, grid)
      if (narrow.lower.every((l, i) => l <= 0 && (narrow.upper[i] ?? 0) >= 0)) coversNull++
    }
    // Whole-curve coverage of the pointwise band is lower than 95%; the simultaneous band is the one that holds it.
    expect(simultaneousNull / reps).toBeGreaterThan(0.85)
    expect(simultaneousNull).toBeGreaterThanOrEqual(coversNull)
    expect(coversTruth / reps).toBeGreaterThan(0.5)
  })
})

describe('partial effects with standard errors', () => {
  const n = 90
  const x = Array.from({ length: n }, (_, i) => [i / (n - 1), i % 3])
  const y = x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + (r[1] ?? 0) + 0.1 * Math.sin(i * 12.9898))
  const fitted = gam.with({ terms: [s(0, { k: 6 }), factor(1, { randomEffect: true })] }).fit({ x, y })

  it('adds up to the linear predictor and its band is the band of the prediction', () => {
    const at = [
      [0.2, 0],
      [0.6, 1],
      [0.9, 2],
    ]
    const all = fitted.predictTermsSe(at)
    const eta = Array.from((fitted.linearPredictor(at) as { data: ArrayLike<number> }).data)
    const intercept = fitted.coef[0] ?? 0
    for (let i = 0; i < 3; i++) {
      expect(intercept + (all.effects[2 * i] ?? 0) + (all.effects[2 * i + 1] ?? 0)).toBeCloseTo(eta[i] ?? 0, 10)
      expect(all.total.fit[i]).toBeCloseTo(eta[i] ?? 0, 10)
    }
    expect(Array.from(all.total.se)).toEqual(Array.from(fitted.predictionBand(at).se))
    expect(all.se.every((v) => v >= 0)).toBe(true)
  })

  it('leaves terms out, including a level never seen in training, and the intercept alone is less certain', () => {
    const all = fitted.predictTermsSe([[0.5, 1]])
    const without = fitted.predictTermsSe([[0.5, 99]], { exclude: [1] })
    expect(without.effects[1]).toBe(0)
    expect(without.se[1]).toBe(0)
    expect(without.effects[0]).toBeCloseTo(all.effects[0] ?? 0, 10)
    // The intercept and the level effects are negatively correlated (they trade off): the intercept alone is far less
    // certain than the intercept plus the level's effect.
    expect(without.total.se[0]).toBeGreaterThan(all.total.se[0] ?? 0)
    expect(() => fitted.predictTermsSe([[0.5, 99]])).toThrow(/level/)
    const ctxBand = predictionBand
    expect(typeof ctxBand).toBe('function')
    expect(typeof predictTermsSe).toBe('function')
  })
})
