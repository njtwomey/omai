import { describe, expect, it } from 'vitest'
import { expectInvariant, expectMoments, gaussian2, moments, standardNormal } from '../../testing/sampling.ts'
import { key, normal } from '../random/index.ts'
import { sampleChains } from './chains.ts'
import { ensembleSliceKernel, stretchMoveKernel } from './ensemble.ts'
import { numericTarget } from './target.ts'

const WALKERS = 8

/** Exact draws of an ensemble of `WALKERS` independent N(0, 1) walkers in one dimension: row-major `[n, WALKERS]`. */
const ensembleDraws = (n: number, seed: number): Float64Array => normal(key(seed), [n * WALKERS]).data

const truth = { mean: new Array<number>(WALKERS).fill(0), sd: new Array<number>(WALKERS).fill(1) }

describe('stretchMoveKernel', () => {
  it('leaves the product of the walkers invariant, with the stretch and the differential move', () => {
    const t = standardNormal(1)
    expectInvariant(stretchMoveKernel(t, { numWalkers: WALKERS }), ensembleDraws(1500, 1), truth, {
      key: key(2),
      steps: 2,
    })
    expectInvariant(
      stretchMoveKernel(t, { numWalkers: WALKERS, move: 'differential' }),
      ensembleDraws(1500, 3),
      truth,
      { key: key(4), steps: 2 },
    )
  })

  it('samples a correlated Gaussian without a proposal to tune, and is affine invariant', () => {
    // Standard deviations 1 and 100: a fixed random walk would need a preconditioner.
    const t = gaussian2([0, 0], [1, 100], 0.9)
    const c = sampleChains(stretchMoveKernel(t, { numWalkers: 12 }), {
      key: key(5),
      numChains: 1,
      numDraws: 3000,
      numWarmup: 300,
      init: (k) => normal(k, [24]).data,
    })
    // Pool the walkers: coordinates of walker w are columns [2w, 2w + 1].
    const x = new Float64Array(12 * 3000)
    const y = new Float64Array(12 * 3000)
    for (let d = 0; d < 3000; d++)
      for (let w = 0; w < 12; w++) {
        x[w * 3000 + d] = c.draws.data[d * 24 + 2 * w] ?? 0
        y[w * 3000 + d] = c.draws.data[d * 24 + 2 * w + 1] ?? 0
      }
    expect(Math.abs(moments(x).sd - 1)).toBeLessThan(0.15)
    expect(Math.abs(moments(y).sd - 100)).toBeLessThan(15)
    expect(c.stats.acceptanceRate?.data.some((v) => v > 0.1)).toBe(true)
  })

  it('validates its settings', () => {
    expect(() => stretchMoveKernel(standardNormal(1), { numWalkers: 2 })).toThrow()
    expect(() => stretchMoveKernel(standardNormal(1), { numWalkers: 6, a: 1 })).toThrow()
    expect(() => stretchMoveKernel(standardNormal(1), { numWalkers: 6 }).init([0, 1])).toThrow()
  })
})

describe('ensembleSliceKernel', () => {
  it('leaves the product of the walkers invariant, with the differential and the Gaussian directions', () => {
    const t = standardNormal(1)
    expectInvariant(ensembleSliceKernel(t, { numWalkers: WALKERS }), ensembleDraws(1200, 6), truth, {
      key: key(7),
      steps: 2,
    })
    expectInvariant(ensembleSliceKernel(t, { numWalkers: WALKERS, move: 'gaussian' }), ensembleDraws(1200, 8), truth, {
      key: key(9),
    })
  })

  it('tunes its scale from the expansions and contractions during warm-up only', () => {
    const t = numericTarget((x) => -0.5 * ((x[0] ?? 0) / 5) ** 2, 1)
    const tuning = ensembleSliceKernel(t, { numWalkers: WALKERS, tuneMu: true, mu: 0.01 })
    let s = tuning.init(normal(key(10), [WALKERS]).data)
    const before = s.mu
    for (let i = 0; i < 30; i++) s = tuning.step(key(20 + i), s).state
    expect(s.mu).not.toBe(before)
    const fixed = ensembleSliceKernel(t, { numWalkers: WALKERS, mu: s.mu })
    let f = fixed.init(s.position)
    f = fixed.step(key(99), f).state
    expect(f.mu).toBe(s.mu)
    expectMoments
  })

  it('validates its settings', () => {
    expect(() => ensembleSliceKernel(standardNormal(1), { numWalkers: 3 })).toThrow()
    expect(() => ensembleSliceKernel(standardNormal(1), { numWalkers: 6, mu: 0 })).toThrow()
  })
})

void expectMoments
