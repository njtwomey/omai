import { describe, expect, it } from 'vitest'
import { bimodal, expectInvariant, expectMoments, gaussian2, moments, standardNormal } from '../../testing/sampling.ts'
import { key, normal } from '../random/index.ts'
import { sampleChain, sampleChains } from './chains.ts'
import { blockedGibbs, ellipticalSlice, sliceSampler } from './gibbs.ts'
import { numericTarget } from './target.ts'

const rho = 0.8
const s = Math.sqrt(1 - rho * rho)
const conditionals = [
  {
    indices: [0],
    sample: (k: Parameters<typeof normal>[0], x: Float64Array) => [
      rho * (x[1] ?? 0) + s * (normal(k, [1]).data[0] ?? 0),
    ],
  },
  {
    indices: [1],
    sample: (k: Parameters<typeof normal>[0], x: Float64Array) => [
      rho * (x[0] ?? 0) + s * (normal(k, [1]).data[0] ?? 0),
    ],
  },
]

describe('blockedGibbs', () => {
  it('samples a correlated Gaussian from its full conditionals, systematic and random scan', () => {
    for (const scan of ['systematic', 'random'] as const) {
      const k = blockedGibbs(2, conditionals, { scan, target: gaussian2([0, 0], [1, 1], rho) })
      const c = sampleChains(k, { key: key(scan === 'random' ? 2 : 1), numChains: 2, numDraws: 2000 })
      expectMoments(c, { mean: [0, 0], sd: [1, 1] })
      expect(Number.isFinite(c.stats.logDensity?.data[0] ?? Number.NaN)).toBe(true)
    }
  })

  it('validates blocks and their draws', () => {
    expect(() => blockedGibbs(2, [])).toThrow()
    expect(() => blockedGibbs(2, [{ indices: [2], sample: () => [0] }])).toThrow()
    const bad = blockedGibbs(2, [{ indices: [0], sample: () => [0, 1] }])
    expect(() => bad.step(key(0), bad.init([0, 0]))).toThrow()
    expect(Number.isNaN(blockedGibbs(1, [{ indices: [0], sample: () => [1] }]).init([0]).logDensity)).toBe(true)
  })
})

describe('sliceSampler', () => {
  it('leaves a standard normal invariant', () => {
    expectInvariant(
      sliceSampler(standardNormal(2), { width: 1.5 }),
      normal(key(3), [2000, 2]).data,
      {
        mean: [0, 0],
        sd: [1, 1],
      },
      { key: key(4) },
    )
  })

  it('samples a bimodal mixture and a heavy-tailed target without tuning', () => {
    const c = sampleChains(sliceSampler(bimodal(2), { width: 2 }), { key: key(5), numChains: 2, numDraws: 2000 })
    expectMoments(c, { mean: [0], sd: [Math.sqrt(1 + 4)] })
    // A Cauchy target: the median is 0 and the quartiles are -1 and 1.
    const cauchy = numericTarget((x) => -Math.log(1 + (x[0] ?? 0) ** 2), 1)
    const r = sampleChain(sliceSampler(cauchy, { width: 1 }), [0], { key: key(6), numDraws: 4000 })
    const sorted = Float64Array.from(r.draws).sort()
    expect(Math.abs(sorted[2000] ?? 0)).toBeLessThan(0.15)
    expect(sorted[1000]).toBeGreaterThan(-1.25)
    expect(sorted[1000]).toBeLessThan(-0.75)
  })

  it('caps its work and stays put on a target that is NaN everywhere', () => {
    const k = sliceSampler(
      numericTarget(() => Number.NaN, 1),
      { maxShrink: 20, maxStepsOut: 3 },
    )
    const t = k.step(key(7), k.init([0.5]))
    expect(t.state.position[0]).toBe(0.5)
    expect(t.info.numSteps).toBeLessThanOrEqual(23)
    expect(() => sliceSampler(standardNormal(1), { width: 0 })).toThrow()
  })
})

describe('ellipticalSlice', () => {
  it('samples a Gaussian posterior from a correlated Gaussian prior and a Gaussian likelihood', () => {
    // Prior N(0, [[1, 0.9], [0.9, 1]]), likelihood N(y = (1, 1); x, I): posterior covariance (P + I)^-1, mean that times y.
    const k = ellipticalSlice(
      (x) => -0.5 * (((x[0] ?? 0) - 1) ** 2 + ((x[1] ?? 0) - 1) ** 2),
      {
        covariance: [
          [1, 0.9],
          [0.9, 1],
        ],
      },
      2,
    )
    const c = sampleChains(k, { key: key(8), numChains: 2, numDraws: 2000 })
    // (P + I)^-1 = S (S + I)^-1 for S the prior covariance: with eigenvalues 1.9 and 0.1 along (1, 1) and (1, -1),
    // the posterior mean is (1.9 / 2.9) (1, 1) and the marginal variance (1.9 / 2.9 + 0.1 / 1.1) / 2.
    const m = 1.9 / 2.9
    const v = (1.9 / 2.9 + 0.1 / 1.1) / 2
    expectMoments(c, { mean: [m, m], sd: [Math.sqrt(v), Math.sqrt(v)] })
  })

  it('accepts a diagonal prior given as a vector or scalar, and keeps the point when every angle fails', () => {
    const k = ellipticalSlice((x) => -0.5 * ((x[0] ?? 0) - 1) ** 2, { covariance: [1], mean: [0] }, 1)
    const r = sampleChain(k, [0], { key: key(9), numDraws: 3000 })
    const { mean, sd } = moments(r.draws)
    expect(mean).toBeCloseTo(0.5, 1)
    expect(sd).toBeCloseTo(Math.sqrt(0.5), 1)
    const stuck = ellipticalSlice(() => Number.NEGATIVE_INFINITY, { covariance: 1 }, 1, { maxIter: 10 })
    const t = stuck.step(key(10), stuck.init([0.3]))
    expect(t.info.accepted).toBe(false)
    expect(t.info.numSteps).toBe(10)
    expect(() =>
      ellipticalSlice(
        () => 0,
        {
          covariance: [
            [1, 2],
            [2, 1],
          ],
        },
        2,
      ),
    ).toThrow()
  })
})
