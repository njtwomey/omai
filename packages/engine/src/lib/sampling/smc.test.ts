import { describe, expect, it } from 'vitest'
import { expectIidMoments } from '../../testing/sampling.ts'
import { multivariateNormal, normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import { malaKernel } from './langevin.ts'
import {
  annealedImportanceSampling,
  nextTemperature,
  resample,
  residualResample,
  smcSampler,
  systematicResample,
} from './smc.ts'
import { numericTarget, proposalFromDistribution } from './target.ts'

/** Counts of each index among resampled indices. */
function counts(idx: Int32Array, k: number): number[] {
  const c = new Array<number>(k).fill(0)
  for (const i of idx) c[i] = (c[i] ?? 0) + 1
  return c
}

describe('resampling', () => {
  const w = [0.1, 0.25, 0.05, 0.6]

  it('keeps every scheme unbiased: mean counts n w_i over many draws', () => {
    for (const scheme of ['multinomial', 'systematic', 'stratified', 'residual'] as const) {
      const total = [0, 0, 0, 0]
      const reps = 400
      for (let r = 0; r < reps; r++) {
        counts(resample(key(r), w, { n: 20, scheme }), 4).forEach((c, i) => {
          total[i] = (total[i] ?? 0) + c
        })
      }
      total.forEach((t, i) => {
        // Each count has variance at most n w (1 - w) <= 5; the mean of 400 is within 4 standard errors.
        expect(Math.abs(t / reps - 20 * (w[i] ?? 0))).toBeLessThan(4 * Math.sqrt(5 / reps))
      })
    }
  })

  it('has low variance for systematic and deterministic floors for residual resampling', () => {
    // Systematic resampling gives floor(n w) or ceil(n w) copies of every index.
    for (let u = 0; u < 1; u += 0.13) {
      counts(systematicResample(w, u, 20), 4).forEach((c, i) => {
        expect(Math.abs(c - 20 * (w[i] ?? 0))).toBeLessThan(1 + 1e-9)
      })
      counts(
        residualResample(
          w,
          Float64Array.from({ length: 21 }, (_, j) => (j + u) / 22),
          20,
        ),
        4,
      ).forEach((c, i) => {
        expect(c).toBeGreaterThanOrEqual(Math.floor(20 * (w[i] ?? 0)))
      })
    }
  })

  it('validates weights and uniforms', () => {
    expect(() => systematicResample([0, 0], 0.5)).toThrow()
    expect(() => systematicResample([-1, 2], 0.5)).toThrow()
    expect(() => resample(key(0), [1, 1], { scheme: 'stratified', n: 3 })).not.toThrow()
  })
})

describe('nextTemperature', () => {
  it('keeps the incremental ESS at the target, and goes to 1 when it can', () => {
    const l = Float64Array.from({ length: 100 }, (_, i) => -0.1 * i)
    const lam = nextTemperature(l, 0, { targetEss: 0.5 })
    const inc = l.map((v) => lam * v)
    let s1 = 0
    let s2 = 0
    for (const v of inc) {
      s1 += Math.exp(v)
      s2 += Math.exp(2 * v)
    }
    expect((s1 * s1) / s2 / 100).toBeCloseTo(0.5, 3)
    expect(nextTemperature([0, -1e-6], 0.5)).toBe(1)
    expect(() => nextTemperature(l, 2)).toThrow()
    expect(() => nextTemperature(l, 0, { targetEss: 1 })).toThrow()
  })
})

describe('smcSampler', () => {
  // Prior N(0, 3^2 I), likelihood N(y = (1, -1); x, I): posterior N(0.9 y, 0.9 I), Z = N(y; 0, 10 I).
  const prior = proposalFromDistribution(
    multivariateNormal([0, 0], {
      covariance: [
        [9, 0],
        [0, 9],
      ],
    }),
  )
  const loglik = (x: Float64Array): number =>
    -0.5 * ((1 - (x[0] ?? 0)) ** 2 + (-1 - (x[1] ?? 0)) ** 2) - Math.log(2 * Math.PI)
  const logZ = -Math.log(2 * Math.PI * 10) - 2 / 20

  it('samples the posterior and estimates the evidence', () => {
    const r = smcSampler(prior, loglik, { key: key(1), numParticles: 400, mcmcSteps: 3 })
    expect(r.converged).toBe(true)
    expect(r.temperatures[0]).toBe(0)
    expect(r.temperatures[r.temperatures.length - 1]).toBe(1)
    expectIidMoments(r.particles.data, 2, { mean: [0.9, -0.9], sd: [Math.sqrt(0.9), Math.sqrt(0.9)] }, 5)
    expect(Math.abs(r.logEvidence - logZ)).toBeLessThan(0.2)
  })

  it('works with every resampling scheme and a gradient-based move, and traces each stage', () => {
    for (const resampling of ['multinomial', 'stratified', 'residual'] as const) {
      const r = smcSampler(prior, loglik, { key: key(2), numParticles: 400, mcmcSteps: 3, resampling })
      expect(Math.abs(r.logEvidence - logZ)).toBeLessThan(0.3)
    }
    const rec = recorder()
    const mala = smcSampler(prior, loglik, {
      key: key(3),
      numParticles: 200,
      mcmcSteps: 2,
      move: (t) =>
        malaKernel(
          numericTarget((x) => t.logDensity(x), 2, {
            grad: (x) => [-(x[0] ?? 0) / 9 + (1 - (x[0] ?? 0)), -(x[1] ?? 0) / 9 + (-1 - (x[1] ?? 0))],
          }),
          { stepSize: 0.3 },
        ),
      tracer: rec,
    })
    expect(mala.converged).toBe(true)
    const temps = metricSeries(rec.run(), 'temperature').values
    expect(temps[temps.length - 1]).toBe(1)
  })

  it('caps the number of stages', () => {
    const sharp = (x: Float64Array): number => -1e6 * ((x[0] ?? 0) - 1) ** 2
    const r = smcSampler(proposalFromDistribution(normal(0, 1)), sharp, {
      key: key(4),
      numParticles: 50,
      mcmcSteps: 1,
      maxStages: 3,
    })
    expect(r.stages).toBe(3)
    expect(r.converged).toBe(false)
  })
})

describe('annealedImportanceSampling', () => {
  it('estimates the evidence of a conjugate model without bias', () => {
    const prior = proposalFromDistribution(normal(0, 3))
    const loglik = (x: Float64Array): number => -0.5 * (1 - (x[0] ?? 0)) ** 2 - 0.5 * Math.log(2 * Math.PI)
    const r = annealedImportanceSampling(prior, loglik, { key: key(5), numParticles: 300 })
    const truth = -0.5 * Math.log(2 * Math.PI * 10) - 0.05
    expect(Math.abs(r.logEvidence.value - truth)).toBeLessThan(4 * (r.logEvidence.standardError ?? 0) + 1e-3)
    expect(r.ess).toBeGreaterThan(100)
    expect(() => annealedImportanceSampling(prior, loglik, { key: key(0), temperatures: [0, 0.5] })).toThrow()
  })
})
