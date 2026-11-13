import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { glmCurve, zeroInflatedCounts } from './glm-curves.ts'

/** The mean of y / mu and the Pearson dispersion mean((y - mu)^2 / V(mu)) for a variance function V. */
function moments(family: string, knobs: Record<string, number | string>, variance: (mu: number) => number) {
  const { data, truth } = generate(glmCurve, { key: key(7), knobs: { n: 20_000, family, ...knobs } })
  const xs = toFlat(data.x)
  const ys = toFlat(data.y)
  const mu = toFlat(truth.regressionFunction?.(data.x) ?? tensor([]))
  let ratio = 0
  let pearson = 0
  for (const [i, y] of ys.entries()) {
    const m = mu[i] ?? 1
    ratio += y / m
    pearson += (y - m) ** 2 / variance(m)
  }
  return { ratio: ratio / ys.length, pearson: pearson / ys.length, xs, ys }
}

describe('glmCurve', () => {
  it('has the mean curve exp(intercept + slope x)', () => {
    const { truth } = generate(glmCurve, { knobs: { intercept: 0.5, slope: -1 } })
    const f = toFlat(truth.regressionFunction?.(tensor([[0], [2]])) ?? tensor([]))
    expect(f[0]).toBeCloseTo(Math.exp(0.5), 12)
    expect(f[1]).toBeCloseTo(Math.exp(-1.5), 12)
  })

  it('draws Poisson and negative binomial counts with their variance functions', () => {
    const p = moments('poisson', {}, (m) => m)
    expect(p.ys.every((y) => Number.isInteger(y) && y >= 0)).toBe(true)
    expect(p.xs.every((x) => x >= 0 && x <= 3)).toBe(true)
    expect(p.ratio).toBeCloseTo(1, 1)
    expect(p.pearson).toBeCloseTo(1, 1)
    const nb = moments('negativeBinomial', { dispersion: 0.5 }, (m) => m + 0.5 * m * m)
    expect(nb.ys.every((y) => Number.isInteger(y) && y >= 0)).toBe(true)
    expect(nb.ratio).toBeCloseTo(1, 1)
    expect(nb.pearson).toBeCloseTo(1, 1)
  })

  it('draws positive gamma and inverse Gaussian responses with variance dispersion mu^2 and mu^3', () => {
    const g = moments('gamma', { dispersion: 0.25 }, (m) => 0.25 * m * m)
    expect(g.ys.every((y) => y > 0)).toBe(true)
    expect(g.ratio).toBeCloseTo(1, 1)
    expect(g.pearson).toBeCloseTo(1, 1)
    const ig = moments('inverseGaussian', { dispersion: 0.1 }, (m) => 0.1 * m ** 3)
    expect(ig.ys.every((y) => y > 0)).toBe(true)
    expect(ig.ratio).toBeCloseTo(1, 1)
    expect(Math.abs(ig.pearson - 1)).toBeLessThan(0.15)
  })

  it('draws Tweedie responses: exact zeros with probability exp(-lambda), variance phi mu^p', () => {
    const phi = 0.8
    const pw = 1.4
    const t = moments('tweedie', { dispersion: phi, power: pw }, (m) => phi * m ** pw)
    expect(t.ys.every((y) => y >= 0)).toBe(true)
    expect(t.ratio).toBeCloseTo(1, 1)
    expect(t.pearson).toBeCloseTo(1, 1)
    const zeros = t.ys.filter((y) => y === 0).length / t.ys.length
    const expected =
      t.xs.reduce((s, x) => s + Math.exp(-(Math.exp(0.8 * x) ** (2 - pw)) / (phi * (2 - pw))), 0) / t.xs.length
    expect(Math.abs(zeros - expected)).toBeLessThan(0.02)
  })
})

describe('zeroInflatedCounts', () => {
  const draw = (knobs: Record<string, number>) => {
    const { data, truth } = generate(zeroInflatedCounts, { key: key(3), knobs: { n: 20_000, ...knobs } })
    return {
      x: toFlat(data.x),
      y: toFlat(data.y),
      structural: toFlat(truth.latents?.['structural'] as never),
      rate: toFlat(truth.latents?.['rate'] as never),
      truth,
      data,
    }
  }

  it('makes every structural zero a zero count, in about the stated share of the rows', () => {
    const d = draw({ inflation: 0.3 })
    expect(d.x.every((x) => x >= -1.5 && x <= 1.5)).toBe(true)
    expect(d.y.every((y) => Number.isInteger(y) && y >= 0)).toBe(true)
    expect(d.structural.every((s, i) => s === 0 || d.y[i] === 0)).toBe(true)
    expect(d.structural.reduce((a, b) => a + b, 0) / d.y.length).toBeCloseTo(0.3, 1)
  })

  it('draws the other rows Poisson at the rate exp(intercept + slope x), zeros included', () => {
    const d = draw({ inflation: 0.3, intercept: 0.5, slope: -0.4 })
    const open = d.y.map((y, i) => [y, d.rate[i] ?? 0] as const).filter((_, i) => d.structural[i] === 0)
    expect(d.rate[0]).toBeCloseTo(Math.exp(0.5 - 0.4 * (d.x[0] ?? 0)), 12)
    const meanRatio = open.reduce((s, [y, r]) => s + y / r, 0) / open.length
    expect(meanRatio).toBeCloseTo(1, 1)
    const zeros = open.filter(([y]) => y === 0).length / open.length
    const expected = open.reduce((s, [, r]) => s + Math.exp(-r), 0) / open.length
    expect(Math.abs(zeros - expected)).toBeLessThan(0.02)
  })

  it('has the mean count (1 - inflation) rate as its regression function, and no structural zeros at zero inflation', () => {
    const d = draw({ inflation: 0, intercept: 0.2, slope: 0.3 })
    const f = toFlat(d.truth.regressionFunction?.(tensor([[0], [1]])) ?? tensor([]))
    expect(f[1]).toBeCloseTo(Math.exp(0.5), 12)
    expect(d.structural.every((s) => s === 0)).toBe(true)
    const half = draw({ inflation: 0.5, intercept: 0.2, slope: 0.3 })
    expect(toFlat(half.truth.regressionFunction?.(tensor([[0]])) ?? tensor([]))[0]).toBeCloseTo(0.5 * Math.exp(0.2), 12)
  })
})
