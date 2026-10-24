import { describe, expect, it } from 'vitest'
import { expectInvariant, expectMoments, gaussian2, standardNormal } from '../../testing/sampling.ts'
import { normal as normalDist, studentT } from '../dists/index.ts'
import { key, normal } from '../random/index.ts'
import { sampleChain, sampleChains } from './chains.ts'
import {
  adaptiveMetropolis,
  independenceMetropolis,
  metropolisAcceptance,
  metropolisHastings,
  randomWalkAdaptation,
  randomWalkMetropolis,
} from './metropolis.ts'
import { banana, numericTarget, proposal, proposalFromDistribution } from './target.ts'

const draws = (n: number, dim: number, seed: number): Float64Array => normal(key(seed), [n, dim]).data

describe('metropolisAcceptance', () => {
  it('is min(1, ratio) and 0 for an undefined ratio', () => {
    expect(metropolisAcceptance(0, -Math.log(2), 0, 0)).toBeCloseTo(0.5, 15)
    expect(metropolisAcceptance(0, 0, Math.log(2), 0)).toBeCloseTo(0.5, 15)
    expect(metropolisAcceptance(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, 0, 0)).toBe(0)
    expect(metropolisAcceptance(Number.NEGATIVE_INFINITY, 0, 0, 0)).toBe(1)
  })
})

describe('randomWalkMetropolis', () => {
  it('leaves a standard normal invariant', () => {
    expectInvariant(
      randomWalkMetropolis(standardNormal(2), { scale: 1.5 }),
      draws(3000, 2, 1),
      {
        mean: [0, 0],
        sd: [1, 1],
      },
      { key: key(2), steps: 2 },
    )
  })

  it('samples a correlated Gaussian with a full proposal covariance', () => {
    const t = gaussian2([1, 2], [1, 3], 0.8)
    const k = randomWalkMetropolis(t, {
      covariance: [
        [1, 2.4],
        [2.4, 9],
      ],
    })
    const c = sampleChains(k, { key: key(3), numChains: 2, numWarmup: 200, numDraws: 2500 })
    expectMoments(c, { mean: [1, 2], sd: [1, 3] })
  })

  it('never moves on a target that is NaN everywhere, and rejects bad scales', () => {
    const t = numericTarget(() => Number.NaN, 1)
    const r = sampleChain(randomWalkMetropolis(t), [0.5], { key: key(4), numDraws: 20 })
    expect(r.draws.every((v) => v === 0.5)).toBe(true)
    expect(() => randomWalkMetropolis(standardNormal(1), { scale: 0 })).toThrow()
    expect(() =>
      randomWalkMetropolis(standardNormal(2), {
        covariance: [
          [1, 2],
          [2, 1],
        ],
      }),
    ).toThrow()
  })
})

describe('independenceMetropolis', () => {
  it('leaves the target invariant with a heavier-tailed proposal and samples it', () => {
    const q = proposalFromDistribution(studentT(4, 0, 1.5))
    const k = independenceMetropolis(standardNormal(1), q)
    expectInvariant(k, draws(3000, 1, 5), { mean: [0], sd: [1] }, { key: key(6) })
    const c = sampleChains(k, { key: key(7), numChains: 2, numDraws: 1500 })
    expectMoments(c, { mean: [0], sd: [1] })
  })

  it('checks dimensions', () => {
    expect(() => independenceMetropolis(standardNormal(2), proposalFromDistribution(normalDist(0, 1)))).toThrow()
  })
})

describe('metropolisHastings', () => {
  it('corrects an asymmetric proposal (a drifting Gaussian walk) with the Hastings ratio', () => {
    // q(x' | x) = N(x'; x + 0.3, 1): biased, so the Hastings correction is essential.
    const logq = (to: number, from: number): number => -0.5 * (to - from - 0.3) ** 2
    const k = metropolisHastings(standardNormal(1), (kk, x) => {
      const y = (x[0] ?? 0) + 0.3 + (normal(kk, [1]).data[0] ?? 0)
      return { proposal: [y], logForward: logq(y, x[0] ?? 0), logBackward: logq(x[0] ?? 0, y) }
    })
    expectInvariant(k, draws(3000, 1, 8), { mean: [0], sd: [1] }, { key: key(9), steps: 2 })
  })

  it('checks the proposal length', () => {
    const k = metropolisHastings(standardNormal(1), () => ({ proposal: [1, 2], logForward: 0, logBackward: 0 }))
    expect(() => k.step(key(0), k.init([0]))).toThrow()
    const q = proposal({ dim: 1, sample: () => [0], logProb: () => 0 })
    expect(q.dim).toBe(1)
  })
})

describe('adaptiveMetropolis', () => {
  it('learns the covariance of a strongly correlated Gaussian and samples it', () => {
    const t = gaussian2([0, 0], [1, 10], 0.95)
    const c = sampleChains(adaptiveMetropolis(t, { initialScale: 0.5 }), {
      key: key(10),
      numChains: 2,
      numWarmup: 1000,
      numDraws: 4000,
    })
    expectMoments(c, { mean: [0, 0], sd: [1, 10] })
    const r = sampleChain(adaptiveMetropolis(t), [0, 0], { key: key(11), numDraws: 4000 })
    const cov01 = (r.state.m2[1] ?? 0) / (r.state.count - 1)
    expect(cov01).toBeGreaterThan(5)
    expect(cov01).toBeLessThan(14)
  })

  it("samples Haario's banana", () => {
    const c = sampleChains(adaptiveMetropolis(banana()), {
      key: key(12),
      numChains: 2,
      numWarmup: 1000,
      numDraws: 4000,
    })
    expectMoments(c, { mean: [0, 0] }, 5)
  })

  it('rejects bad settings', () => {
    expect(() => adaptiveMetropolis(standardNormal(1), { initialScale: 0 })).toThrow()
  })
})

describe('randomWalkAdaptation', () => {
  it('tunes the scale towards an acceptance rate of 0.234 and learns the shape', () => {
    const t = gaussian2([0, 0], [1, 5], 0.9)
    const c = sampleChains(randomWalkAdaptation(t), { key: key(13), numChains: 2, numWarmup: 1500, numDraws: 2000 })
    const a = c.stats.acceptanceRate?.data ?? new Float64Array()
    const mean = a.reduce((x, y) => x + y, 0) / a.length
    expect(mean).toBeGreaterThan(0.15)
    expect(mean).toBeLessThan(0.4)
    expect((c.parameters[0] as { shapeFactor?: number[] }).shapeFactor?.length).toBe(4)
    expectMoments(c, { mean: [0, 0], sd: [1, 5] })
    expect(() => randomWalkAdaptation(t, { targetAcceptance: 1 })).toThrow()
  })
})
