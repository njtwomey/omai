import { describe, expect, it } from 'vitest'
import { bayesPredict, generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { biasedCredit, germanCredit, titanicSurvival } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

/** The first three columns of four. */
function withoutGroup(x: ArrayLike<number>): Tensor {
  const n = x.length / 4
  const out = new Float64Array(3 * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) out[3 * i + j] = x[4 * i + j] ?? 0
  return fromStorage(out, [n, 3], 'float64')
}

/** The share of label 1 among the rows of each group. */
function baseRates(y: readonly number[], group: readonly number[]): number[] {
  return [0, 1].map((a) => mean(y.filter((_, i) => group[i] === a)))
}

describe('biasedCredit', () => {
  const big = { n: 40_000 }

  it('draws the minority with the asked share and puts the group in the last column', () => {
    const { data, truth } = generate(biasedCredit, { key: key(1), knobs: { ...big, minorityShare: 0.2 } })
    const group = truth.latents?.['group'] as number[]
    expect(mean(group)).toBeCloseTo(0.2, 2)
    const x = toFlat(data.x)
    expect(group.every((a, i) => x[4 * i + 3] === a)).toBe(true)
    expect(data.meta.sensitive).toEqual({ group: ['majority', 'minority'] })
    expect(data.classes).toEqual(['default', 'repaid'])
  })

  it('records the base rates the truth states, the label bias thinning the minority’s', () => {
    const knobs = { ...big, baseRateGap: 0.6, labelBias: 0.4 }
    const { data, truth } = generate(biasedCredit, { key: key(2), knobs })
    const group = truth.latents?.['group'] as number[]
    const clean = truth.latents?.['cleanLabels'] as number[]
    const recorded = truth.extra?.['recordedBaseRates'] as number[]
    const cleanRates = truth.extra?.['cleanBaseRates'] as number[]
    const y = Array.from(toFlat(data.y))
    for (const a of [0, 1]) {
      expect(baseRates(y, group)[a]).toBeCloseTo(recorded[a] ?? 0, 1)
      expect(baseRates(clean, group)[a]).toBeCloseTo(cleanRates[a] ?? 0, 1)
    }
    // The majority's labels are clean; only the minority's good outcomes were dropped.
    expect(y.every((v, i) => group[i] === 1 || v === clean[i])).toBe(true)
    expect(y.every((v, i) => v <= (clean[i] ?? 0))).toBe(true)
    expect((recorded[1] ?? 0) / (cleanRates[1] ?? 1)).toBeCloseTo(0.6, 12)
  })

  it('without bias or gap the groups share a base rate', () => {
    const { data, truth } = generate(biasedCredit, { key: key(3), knobs: { ...big, baseRateGap: 0, labelBias: 0 } })
    const r = baseRates(Array.from(toFlat(data.y)), truth.latents?.['group'] as number[])
    expect(Math.abs((r[0] ?? 0) - (r[1] ?? 0))).toBeLessThan(0.03)
  })

  it('the proxy separates the groups by its strength', () => {
    for (const strength of [0, 2]) {
      const { data, truth } = generate(biasedCredit, { key: key(4), knobs: { n: 20_000, proxyStrength: strength } })
      const group = truth.latents?.['group'] as number[]
      const x = toFlat(data.x)
      const postcode = group.map((_, i) => x[4 * i + 2] ?? 0)
      const gap = mean(postcode.filter((_, i) => group[i] === 1)) - mean(postcode.filter((_, i) => group[i] === 0))
      expect(gap).toBeCloseTo(strength, 1)
    }
  })

  it('the minority’s measurements are noisier by the asked factor', () => {
    const { data, truth } = generate(biasedCredit, { key: key(5), knobs: { ...big, minorityNoise: 2 } })
    const group = truth.latents?.['group'] as number[]
    const q = truth.latents?.['qualification'] as number[]
    const x = toFlat(data.x)
    const sd = (a: number): number => {
      const r = group.flatMap((g, i) => (g === a ? [(x[4 * i] ?? 0) - (q[i] ?? 0)] : []))
      return Math.sqrt(mean(r.map((v) => v * v)))
    }
    expect(sd(0)).toBeCloseTo(1, 1)
    expect(sd(1)).toBeCloseTo(2, 1)
  })

  it("the Bayes rule's error matches the stated Bayes error; the posterior without the group is calibrated", () => {
    const knobs = { n: 20_000, proxyStrength: 2, labelBias: 0.3 }
    const { data, truth } = generate(biasedCredit, { key: key(6), knobs })
    expect(data.x.shape).toEqual([20_000, 4])
    const p = toFlat(truth.posterior?.(data.x) as Tensor)
    for (let i = 0; i < 50; i++) expect((p[2 * i] ?? 0) + (p[2 * i + 1] ?? 0)).toBeCloseTo(1, 12)
    const y = Array.from(toFlat(data.y))
    const err = mean(Array.from(toFlat(bayesPredict(truth, data.x))).map((v, i) => (v === y[i] ? 0 : 1)))
    const b = truth.bayesError ?? Number.NaN
    expect(Math.abs(err - b)).toBeLessThan(5 * Math.sqrt((b * (1 - b)) / 20_000) + 0.005)
    // Calibrated: the posterior's mean matches the label's mean, with or without the group.
    expect(mean(Array.from({ length: 20_000 }, (_, i) => p[2 * i + 1] ?? 0))).toBeCloseTo(mean(y), 2)
    const unaware = truth.extra?.['unawarePosterior'] as (x: Tensor) => Tensor
    const three = withoutGroup(toFlat(data.x))
    const q = toFlat(unaware(three))
    expect(mean(Array.from({ length: 20_000 }, (_, i) => q[2 * i + 1] ?? 0))).toBeCloseTo(mean(y), 2)
    // Without the group, the best predictor errs more.
    const unawareErr = mean(y.map((v, i) => (((q[2 * i + 1] ?? 0) > 0.5 ? 1 : 0) === v ? 0 : 1)))
    expect(unawareErr).toBeGreaterThan(err)
  })

  it('is a pure function of its key', () => {
    const a = generate(biasedCredit, { key: key(7), knobs: { n: 50 } }).data
    const b = generate(biasedCredit, { key: key(7), knobs: { n: 50 } }).data
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
    expect(toFlat(a.y)).toEqual(toFlat(b.y))
  })
})

describe('vendored data with a protected attribute', () => {
  it('German credit: 1000 applicants, 700 good, sex as the protected attribute', () => {
    const d = germanCredit()
    expect(d.x.shape).toEqual([1000, 12])
    expect(toFlat(d.y).filter((v) => v === 1).length).toBe(700)
    expect(d.meta.sensitive).toEqual({ female: ['male', 'female'] })
    const x = toFlat(d.x)
    expect(Array.from({ length: 1000 }, (_, i) => x[12 * i + 11] ?? 0).filter((v) => v === 1).length).toBe(310)
  })

  it('Titanic survival: 2201 people, 711 survivors, women surviving far more often', () => {
    const d = titanicSurvival()
    expect(d.x.shape).toEqual([2201, 5])
    const y = Array.from(toFlat(d.y))
    expect(y.filter((v) => v === 1).length).toBe(711)
    const x = toFlat(d.x)
    const female = y.map((_, i) => x[5 * i + 4] ?? 0)
    const r = baseRates(y, female)
    expect(r[1]).toBeCloseTo(344 / 470, 12)
    expect(r[0]).toBeCloseTo(367 / 1731, 12)
  })
})
