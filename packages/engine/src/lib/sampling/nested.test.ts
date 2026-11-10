// Nested sampling against analytic evidences: a Gaussian likelihood under a Gaussian prior (Z = (2 pi s^2)^(d/2) N(y; 0,
// (t^2 + s^2) I)) and under a uniform prior on a box (Z = (2 pi s^2)^(d/2) / (2a)^d), in one to five dimensions, within the
// error the run states; the posterior weights give the exact posterior moments; the error bar is calibrated (the draws of
// log Z that carry the volume uncertainty have the spread the state reports, and the stated error covers the truth over
// seeds); and the run is steppable (a serialised state resumes to the same run).
import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { nestedAlgorithm, nestedEvidence, nestedLogZDraws, nestedPosterior, nestedSampler } from './nested.ts'
import { proposal } from './target.ts'

const PRIOR_SD = 3
const SIGMA = 0.5

// Likelihood exp(-|x - y|^2 / (2 sigma^2)), unnormalised; prior N(0, prior_sd^2 I) on each coordinate.
const gaussianTruth = (y: readonly number[]): number => {
  const d = y.length
  const v = PRIOR_SD ** 2 + SIGMA ** 2
  const quad = y.reduce((a, b) => a + b * b, 0) / v
  return 0.5 * d * Math.log(2 * Math.PI * SIGMA ** 2) - 0.5 * d * Math.log(2 * Math.PI * v) - 0.5 * quad
}

function gaussianProblem(y: readonly number[]) {
  const d = y.length
  const prior = proposal({
    dim: d,
    sample: (k) => normal(k, [d]).data.map((v) => v * PRIOR_SD),
    logProb: (x) =>
      Array.from(x).reduce((a, v) => a - 0.5 * (v / PRIOR_SD) ** 2 - Math.log(PRIOR_SD * Math.sqrt(2 * Math.PI)), 0),
  })
  const logL = (x: Float64Array): number =>
    (-0.5 * Array.from(x).reduce((a, v, i) => a + (v - (y[i] ?? 0)) ** 2, 0)) / SIGMA ** 2
  return { prior, logL }
}

describe('nested sampling evidence', () => {
  it('matches the analytic Gaussian evidence in one, two and five dimensions within the stated error', () => {
    for (const [seed, y, numLive] of [
      [1, [1.2], 150],
      [2, [0.8, -1.1], 150],
      [3, [0.5, -0.4, 0.9, 0.2, -0.7], 120],
    ] as const) {
      const { prior, logL } = gaussianProblem(y)
      const r = nestedSampler(prior, logL, { key: key(seed), numLive })
      expect(r.converged).toBe(true)
      // Nested sampling's error is the dominant one: the truth is within four of its standard errors.
      expect(Math.abs(r.logZ - gaussianTruth(y)), `d = ${y.length}`).toBeLessThan(4 * r.logZError + 0.05)
      expect(r.logZError).toBeGreaterThan(0)
      expect(r.logZError).toBeLessThan(0.6)
    }
  })

  it('matches the evidence of a Gaussian likelihood under a uniform prior on a box', () => {
    const a = 6
    const prior = proposal({
      dim: 2,
      sample: (k) => uniform(k, [2], { low: -a, high: a }).data,
      logProb: (x) => (Array.from(x).every((v) => Math.abs(v) < a) ? -2 * Math.log(2 * a) : Number.NEGATIVE_INFINITY),
    })
    const y = [0.7, -0.3]
    const logL = (x: Float64Array): number =>
      (-0.5 * ((x[0] ?? 0) - y[0]!) ** 2) / SIGMA ** 2 - (0.5 * ((x[1] ?? 0) - y[1]!) ** 2) / SIGMA ** 2
    const truth = Math.log(2 * Math.PI * SIGMA ** 2) - 2 * Math.log(2 * a)
    const r = nestedSampler(prior, logL, { key: key(4), numLive: 150 })
    expect(Math.abs(r.logZ - truth)).toBeLessThan(4 * r.logZError + 0.05)
  })

  it('the random-walk inner sampler gives the same evidence', () => {
    const y = [0.8, -1.1]
    const { prior, logL } = gaussianProblem(y)
    const r = nestedSampler(prior, logL, { key: key(5), numLive: 150, inner: 'random-walk', innerSteps: 20 })
    expect(Math.abs(r.logZ - gaussianTruth(y))).toBeLessThan(4 * r.logZError + 0.1)
  })

  it('the stated error covers the truth over seeds, and the volume-draw spread agrees with it', () => {
    const y = [1.5]
    const { prior, logL } = gaussianProblem(y)
    const truth = gaussianTruth(y)
    const z: number[] = []
    let covered = 0
    const runs = 12
    for (let seed = 0; seed < runs; seed++) {
      const r = nestedSampler(prior, logL, { key: key(100 + seed), numLive: 100 })
      z.push(r.logZ)
      if (Math.abs(r.logZ - truth) < 2 * r.logZError) covered += 1
      if (seed === 0) {
        const draws = nestedLogZDraws(r.state, 1, 100, key(9), 400)
        const m = draws.reduce((p, q) => p + q, 0) / draws.length
        const sd = Math.sqrt(draws.reduce((p, q) => p + (q - m) ** 2, 0) / (draws.length - 1))
        expect(sd / r.logZError).toBeGreaterThan(0.6)
        expect(sd / r.logZError).toBeLessThan(1.6)
      }
    }
    expect(covered).toBeGreaterThanOrEqual(9)
    const m = z.reduce((a, b) => a + b, 0) / runs
    expect(Math.abs(m - truth)).toBeLessThan(0.15)
  })

  it('the weighted points have the exact posterior moments', () => {
    const y = [1.2, -0.6]
    const { prior, logL } = gaussianProblem(y)
    const r = nestedSampler(prior, logL, { key: key(6), numLive: 300 })
    const post = nestedPosterior(r.state, 2, 300)
    expect(post.logWeights.reduce((a, v) => a + Math.exp(v), 0)).toBeCloseTo(1, 10)
    const k = PRIOR_SD ** 2 / (PRIOR_SD ** 2 + SIGMA ** 2)
    const sd = Math.sqrt((PRIOR_SD ** 2 * SIGMA ** 2) / (PRIOR_SD ** 2 + SIGMA ** 2))
    for (let a = 0; a < 2; a++) {
      let mean = 0
      for (let i = 0; i < post.logWeights.length; i++)
        mean += Math.exp(post.logWeights[i] ?? 0) * (post.points[i * 2 + a] ?? 0)
      let v = 0
      for (let i = 0; i < post.logWeights.length; i++)
        v += Math.exp(post.logWeights[i] ?? 0) * ((post.points[i * 2 + a] ?? 0) - mean) ** 2
      expect(Math.abs(mean - k * (y[a] ?? 0))).toBeLessThan(0.08)
      expect(Math.abs(Math.sqrt(v) / sd - 1)).toBeLessThan(0.15)
    }
  })
})

describe('nested sampling as an algorithm', () => {
  const y = [0.9]
  const { prior, logL } = gaussianProblem(y)
  const alg = nestedAlgorithm(prior, logL, { numLive: 40, tolerance: 0.1 })

  it('is deterministic in its key and a serialised state resumes to the same run', () => {
    const whole = runAlgorithm(alg, { seed: 7, maxSteps: 200 })
    const first = runAlgorithm(alg, { seed: 7, maxSteps: 60 })
    expect(first.status).not.toBe('converged')
    const restored = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(first.state))))
    const resumed = runAlgorithm(alg, { seed: 7, maxSteps: 200, start: { state: restored, step: first.steps } })
    expect(resumed.steps).toBe(whole.steps)
    expect(resumed.state.logZDead).toBe(whole.state.logZDead)
    expect(Array.from(resumed.state.live)).toEqual(Array.from(whole.state.live))
    expect(nestedEvidence(resumed.state, 40).logZ).toBe(nestedEvidence(whole.state, 40).logZ)
  })

  it('has an evidence estimate at every step (anytime) that approaches the truth, and logs the contract paths', () => {
    const logs: Record<string, number[]> = {}
    const wrapped = {
      ...alg,
      step: (s: never, ctx: Parameters<typeof alg.step>[1]) =>
        alg.step(s, {
          ...ctx,
          log: (path: string, v: unknown) => {
            if (typeof v === 'number') (logs[path] ??= []).push(v)
          },
          wants: () => true,
        } as never),
    }
    runAlgorithm(wrapped as never, { seed: 3, maxSteps: 400 })
    for (const p of ['ns/logZ', 'ns/logZError', 'ns/logVolume', 'ns/logL', 'ns/information', 'ns/evaluations'])
      expect(logs[p]?.length, p).toBeGreaterThan(10)
    // The estimate including the live points is finite from the first step and the volume only shrinks.
    expect((logs['ns/logZ'] ?? []).every(Number.isFinite)).toBe(true)
    const vol = logs['ns/logVolume'] ?? []
    expect(vol.every((v, i) => i === 0 || v < (vol[i - 1] ?? 0))).toBe(true)
    expect(Math.abs((logs['ns/logZ'] ?? []).at(-1)! - gaussianTruth(y))).toBeLessThan(0.7)
  })

  it('rejects bad options', () => {
    expect(() => nestedAlgorithm(prior, logL, { numLive: 1 })).toThrow(/live points/)
    expect(() => nestedAlgorithm(prior, logL, { tolerance: 0 })).toThrow(/tolerance/)
    expect(() => nestedAlgorithm(prior, logL, { innerSteps: 0 })).toThrow(/innerSteps/)
  })
})
