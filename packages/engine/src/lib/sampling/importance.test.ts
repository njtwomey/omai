import { describe, expect, it } from 'vitest'
import { normal as normalDist } from '../dists/index.ts'
import { key, normal } from '../random/index.ts'
import {
  importanceEss,
  importanceEstimate,
  importanceSampling,
  logEvidence,
  normaliseLogWeights,
  paretoKHat,
  paretoSmoothedWeights,
  selfNormalisedEstimate,
  weightDiagnostics,
} from './importance.ts'
import { numericTarget, proposalFromDistribution } from './target.ts'

describe('importance weights', () => {
  it('normalise stably and give Kish ESS between 1 and n', () => {
    const w = normaliseLogWeights([1000, 1000, Number.NaN])
    expect(w[0]).toBeCloseTo(0.5, 12)
    expect(w[1]).toBeCloseTo(0.5, 12)
    expect(w[2]).toBe(0)
    expect(importanceEss([0, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY])).toBe(1)
    expect(() => normaliseLogWeights([])).toThrow()
  })

  it('give self-normalised estimates with a standard error that covers the truth', () => {
    // pi = N(1, 1) (unnormalised), q = N(0, 2): E_pi[x] = 1, E_pi[x^2] = 2.
    const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) - 1) ** 2, 1)
    const r = importanceSampling(t, proposalFromDistribution(normalDist(0, 2)), { key: key(1), n: 5000 })
    const x = r.draws.data
    const m = selfNormalisedEstimate(x, r.logWeights)
    expect(Math.abs(m.value - 1)).toBeLessThan(4 * (m.standardError ?? 0))
    const m2 = selfNormalisedEstimate(
      x.map((v) => v * v),
      r.logWeights,
    )
    expect(Math.abs(m2.value - 2)).toBeLessThan(4 * (m2.standardError ?? 0))
    expect(m.ess).toBeCloseTo(r.ess, 8)
    expect(r.ess).toBeGreaterThan(1500)
    expect(() => selfNormalisedEstimate([1, 2], [0])).toThrow()
  })

  it('estimate the evidence with a standard error', () => {
    const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2) + 3, 2)
    // q = N(0, 1.5^2 I), written out with batched draws and densities.
    const r = importanceSampling(
      t,
      {
        dim: 2,
        sample: () => new Float64Array(2),
        logProb: () => 0,
        sampleMany: (k, n) => normal(k, [n, 2], { scale: 1.5 }).data,
        logProbMany: (xs) =>
          Float64Array.from({ length: xs.length / 2 }, (_, i) => {
            const a = xs[2 * i] ?? 0
            const b = xs[2 * i + 1] ?? 0
            return -(a * a + b * b) / (2 * 2.25) - Math.log(2 * Math.PI * 2.25)
          }),
      },
      { key: key(2), n: 4000 },
    )
    // Z = e^3 2 pi.
    const truth = 3 + Math.log(2 * Math.PI)
    expect(Math.abs(r.logEvidence.value - truth)).toBeLessThan(4 * (r.logEvidence.standardError ?? 0))
    expect(logEvidence([0]).standardError).toBeNaN()
  })

  it('checks dimensions and sizes', () => {
    const t = numericTarget(() => 0, 2)
    expect(() => importanceSampling(t, proposalFromDistribution(normalDist(0, 1)), { key: key(0), n: 10 })).toThrow()
    expect(() =>
      importanceSampling(
        numericTarget(() => 0, 1),
        proposalFromDistribution(normalDist(0, 1)),
        { key: key(0), n: 0 },
      ),
    ).toThrow()
  })
})

describe('the ordinary importance estimate', () => {
  it('averages the weighted values and reports sd(w f) / sqrt(n) as its error', () => {
    const e = importanceEstimate([1, 2, 3], [0, 0, 0])
    expect(e.value).toBe(2)
    expect(e.standardError).toBeCloseTo(1 / Math.sqrt(3), 12)
    expect(importanceEstimate([1, 1], [Math.log(2), 0]).value).toBeCloseTo(1.5, 12)
  })

  it('recovers E[x^2] = 1 under N(0, 1) from a wider proposal, within its own error', () => {
    const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * Math.log(2 * Math.PI), 1)
    const r = importanceSampling(t, proposalFromDistribution(normalDist(0, 2)), { key: key(3), n: 4000 })
    const f = Array.from(r.draws.data, (x) => x * x)
    const e = importanceEstimate(f, r.logWeights)
    expect(Math.abs(e.value - 1)).toBeLessThan(4 * (e.standardError ?? 0))
  })

  it('refuses mismatched lengths and a single draw', () => {
    expect(() => importanceEstimate([1, 2], [0])).toThrow(/2 values for 1 weights/)
    expect(() => importanceEstimate([1], [0])).toThrow(/at least two/)
  })
})

describe('Pareto smoothing and diagnostics', () => {
  it('flags a proposal that is too narrow (heavy-tailed weights) and passes a wide one', () => {
    const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
    const narrow = importanceSampling(t, proposalFromDistribution(normalDist(0, 0.4)), { key: key(3), n: 2000 })
    const wide = importanceSampling(t, proposalFromDistribution(normalDist(0, 1.5)), { key: key(4), n: 2000 })
    const bad = weightDiagnostics(narrow.logWeights)
    const good = weightDiagnostics(wide.logWeights)
    expect(bad.paretoK).toBeGreaterThan(0.7)
    expect(bad.reliable).toBe(false)
    expect(good.paretoK).toBeLessThan(0.5)
    expect(good.reliable).toBe(true)
    expect(good.essFraction).toBeGreaterThan(bad.essFraction)
    expect(good.minSampleSize).toBeLessThan(bad.minSampleSize)
  })

  it('smooths the largest weights, keeps their order and normalises', () => {
    const lr = normal(key(5), [500], { scale: 1.2 }).data
    const s = paretoSmoothedWeights(lr)
    const w = s.logWeights.data
    let total = 0
    for (const v of w) total += Math.exp(v)
    expect(total).toBeCloseTo(1, 12)
    // The largest raw ratio still has the largest smoothed weight.
    let iMax = 0
    for (let i = 1; i < lr.length; i++) if ((lr[i] ?? 0) > (lr[iMax] ?? 0)) iMax = i
    for (const v of w) expect(v).toBeLessThanOrEqual((w[iMax] ?? 0) + 1e-12)
    expect(() => paretoSmoothedWeights([1, 2, 3])).toThrow()
    expect(() => paretoKHat([1, 2, 3])).toThrow()
    expect(weightDiagnostics([0, 1, 2]).paretoK).toBeNaN()
  })
})
