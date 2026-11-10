// The evidence estimators against closed forms: bridge sampling on exact draws of a Gaussian and of a skewed target (a
// log-gamma), thermodynamic integration and the stepping-stone estimator on exact power posteriors of a Gaussian model, the
// fitted Gaussian proposal's density, and the harmonic mean, which is refused and, when asked for, shown to fail.
import { describe, expect, it } from 'vitest'
import { lgamma } from '../numerics/special/index.ts'
import { gamma as drawGamma, foldIn, key, normal } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import {
  bridgeSampling,
  bridgeSamplingFromRatios,
  fitGaussian,
  gaussianProposal,
  harmonicMeanEvidence,
  steppingStone,
  thermodynamicIntegration,
} from './evidence.ts'
import { proposal } from './target.ts'

describe('bridge sampling', () => {
  it('recovers the normalising constant of a three-dimensional Gaussian within its error', () => {
    // p~(x) = exp(-0.5 x' A x) with A = diag(1, 4, 0.25): Z = (2 pi)^(3/2) / sqrt(det A) = (2 pi)^(3/2).
    const A = [1, 4, 0.25]
    const sd = A.map((a) => 1 / Math.sqrt(a))
    const truth = 1.5 * Math.log(2 * Math.PI) - 0.5 * Math.log(1 * 4 * 0.25)
    const z = normal(key(1), [3 * 600]).data
    const draws = Float64Array.from(z, (v, i) => v * (sd[i % 3] ?? 1))
    const logTarget = (x: Float64Array): number => -0.5 * A.reduce((s, a, i) => s + a * (x[i] ?? 0) ** 2, 0)
    const r = bridgeSampling(logTarget, draws, 3, { key: key(2) })
    expect(r.converged).toBe(true)
    expect(Math.abs(r.logEvidence - truth)).toBeLessThan(4 * r.standardError + 0.01)
    expect(r.standardError).toBeLessThan(0.05)
  })

  it('handles a skewed posterior (a log-gamma) where a Gaussian proposal is only approximate', () => {
    // x = log g with g ~ Gamma(a, 1): p~(x) = exp(a x - e^x), Z = Gamma(a).
    const a = 4
    const truth = item(lgamma(a))
    const draws = Float64Array.from(drawGamma(key(3), [800], a).data, Math.log)
    const r = bridgeSampling((x) => a * (x[0] ?? 0) - Math.exp(x[0] ?? 0), draws, 1, { key: key(4) })
    expect(Math.abs(r.logEvidence - truth)).toBeLessThan(4 * r.standardError + 0.02)
  })

  it('has an error that covers the truth over replications', () => {
    const truth = 0.5 * Math.log(2 * Math.PI)
    let covered = 0
    const reps = 20
    for (let i = 0; i < reps; i++) {
      const draws = normal(foldIn(key(5), i), [200]).data
      const r = bridgeSampling((x) => -0.5 * (x[0] ?? 0) ** 2, draws, 1, { key: foldIn(key(6), i) })
      if (Math.abs(r.logEvidence - truth) < 2.5 * r.standardError) covered += 1
    }
    expect(covered).toBeGreaterThanOrEqual(17)
  })

  it('needs enough draws, and the ratio form is exact when the proposal is the target', () => {
    expect(() => bridgeSampling(() => 0, new Float64Array(4), 1, { key: key(0) })).toThrow(/eight/)
    expect(bridgeSamplingFromRatios([0, 0, 0, 0], [0, 0, 0, 0]).logEvidence).toBeCloseTo(0, 12)
    expect(() => bridgeSamplingFromRatios([0], [0, 0])).toThrow(/two draws/)
  })
})

describe('the fitted Gaussian proposal', () => {
  it('fits weighted draws and has the Gaussian log density, with a defensive prior mixed in', () => {
    const fit = fitGaussian(Float64Array.of(0, 2, 4, 6), 1)
    expect(fit.mean[0]).toBe(3)
    expect(fit.covariance[0]).toBeCloseTo(20 / 3, 12)
    const weighted = fitGaussian(Float64Array.of(0, 10), 1, [Math.log(3), Math.log(1)])
    expect(weighted.mean[0]).toBeCloseTo(2.5, 12)
    const q = gaussianProposal({ mean: Float64Array.of(1, 0), covariance: Float64Array.of(4, 0, 0, 1) })
    expect(q.logProb([1, 0])).toBeCloseTo(-Math.log(2 * Math.PI) - Math.log(2), 12)
    const prior = proposal({ dim: 2, sample: () => new Float64Array(2), logProb: () => -2 })
    const mixed = gaussianProposal(
      { mean: Float64Array.of(1, 0), covariance: Float64Array.of(4, 0, 0, 1) },
      { defensive: { prior, weight: 0.25 } },
    )
    const expected = Math.log(0.75 * Math.exp(q.logProb([1, 0])) + 0.25 * Math.exp(-2))
    expect(mixed.logProb([1, 0])).toBeCloseTo(expected, 12)
    expect(() =>
      gaussianProposal({ mean: Float64Array.of(0), covariance: Float64Array.of(1) }, { inflate: 0 }),
    ).toThrow()
  })
})

describe('thermodynamic integration and the stepping-stone estimator on exact power posteriors', () => {
  // Prior N(0, s0^2), L(theta) = exp(-(y - theta)^2 / (2 s^2)): the power posterior at beta is N(m_b, v_b) with
  // 1 / v_b = 1 / s0^2 + beta / s^2 and m_b = v_b beta y / s^2; Z = sqrt(2 pi s^2) N(y; 0, s0^2 + s^2).
  const s0 = 3
  const s = 0.5
  const y = 1.3
  const truth =
    0.5 * Math.log(2 * Math.PI * s * s) -
    0.5 * Math.log(2 * Math.PI * (s0 * s0 + s * s)) -
    (y * y) / (2 * (s0 * s0 + s * s))
  const rungs = (K: number, n: number): { beta: number; logLikelihoods: Float64Array }[] =>
    Array.from({ length: K + 1 }, (_, k) => {
      const beta = (k / K) ** 4
      const v = 1 / (1 / (s0 * s0) + beta / (s * s))
      const m = (v * beta * y) / (s * s)
      const z = normal(foldIn(key(7), k), [n]).data
      return { beta, logLikelihoods: Float64Array.from(z, (e) => (-0.5 * (y - (m + Math.sqrt(v) * e)) ** 2) / (s * s)) }
    })

  it('both recover the evidence of a Gaussian model, the stepping stone more closely', () => {
    const ladder = rungs(60, 1000)
    const ss = steppingStone(ladder)
    const ti = thermodynamicIntegration(ladder)
    expect(Math.abs(ss.value - truth)).toBeLessThan(4 * ss.standardError + 0.005)
    expect(Math.abs(ti.value - truth)).toBeLessThan(0.05)
    expect(ss.standardError).toBeGreaterThan(0)
    expect(ti.standardError).toBeGreaterThan(0)
    expect(ss.method).toBe('stepping stone')
  })

  it('thermodynamic integration converges with the ladder, and bad ladders are refused', () => {
    const coarse = Math.abs(thermodynamicIntegration(rungs(6, 2000)).value - truth)
    const fine = Math.abs(thermodynamicIntegration(rungs(80, 2000)).value - truth)
    expect(fine).toBeLessThan(coarse)
    expect(() => steppingStone([{ beta: 0, logLikelihoods: [0] }])).toThrow(/two rungs/)
    expect(() =>
      thermodynamicIntegration([
        { beta: 0.5, logLikelihoods: [0, 0] },
        { beta: 0.2, logLikelihoods: [0, 0] },
      ]),
    ).toThrow(/increase/)
  })
})

describe('the harmonic mean estimator', () => {
  it('is refused unless the infinite variance is acknowledged', () => {
    expect(() => harmonicMeanEvidence([-1, -2, -3])).toThrow(/infinite variance/)
  })

  it('fails when shown: a diffuse prior over a sharp likelihood makes it vary far more than bridge sampling', () => {
    // Prior N(0, 10^2), L = exp(-theta^2 / (2 * 0.1^2)) at y = 0: the posterior is N(0, ~0.1^2), the draws are exact.
    const s0 = 10
    const s = 0.1
    const v = 1 / (1 / (s0 * s0) + 1 / (s * s))
    const truth = 0.5 * Math.log(2 * Math.PI * s * s) - 0.5 * Math.log(2 * Math.PI * (s0 * s0 + s * s))
    const harmonic: number[] = []
    const bridge: number[] = []
    for (let i = 0; i < 20; i++) {
      const draws = Float64Array.from(normal(foldIn(key(8), i), [300]).data, (e) => Math.sqrt(v) * e)
      harmonic.push(
        harmonicMeanEvidence(
          Array.from(draws, (t) => (-0.5 * t * t) / (s * s)),
          { acknowledgeInfiniteVariance: true },
        ).value,
      )
      const logTarget = (x: Float64Array): number =>
        (-0.5 * (x[0] ?? 0) ** 2) / (s * s) - 0.5 * ((x[0] ?? 0) / s0) ** 2 - Math.log(s0 * Math.sqrt(2 * Math.PI))
      bridge.push(bridgeSampling(logTarget, draws, 1, { key: foldIn(key(9), i) }).logEvidence)
    }
    const rms = (v: number[]): number => Math.sqrt(v.reduce((a, b) => a + (b - truth) ** 2, 0) / v.length)
    expect(rms(bridge)).toBeLessThan(0.05)
    expect(rms(harmonic)).toBeGreaterThan(5 * rms(bridge))
  })
})
