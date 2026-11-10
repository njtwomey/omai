import { describe, expect, it } from 'vitest'
import { expectInvariant, expectMoments, gaussian2, standardNormal } from '../../testing/sampling.ts'
import { key, normal, uniform } from '../random/index.ts'
import { barkerKernel } from './barker.ts'
import { sampleChain, sampleChains } from './chains.ts'
import { numericTarget } from './target.ts'

/** Exact draws of a correlated bivariate Gaussian with sds (1, 2) and correlation 0.7, row-major `[n, 2]`. */
function gaussianDraws(n: number, seed: number): Float64Array {
  const z = normal(key(seed), [n, 2]).data
  return Float64Array.from({ length: 2 * n }, (_, k) => {
    const i = Math.floor(k / 2)
    const a = z[2 * i] ?? 0
    return k % 2 === 0 ? a : 2 * (0.7 * a + Math.sqrt(1 - 0.49) * (z[2 * i + 1] ?? 0))
  })
}

describe('barkerKernel', () => {
  it('leaves a correlated Gaussian invariant, with and without a preconditioner', () => {
    const t = gaussian2([0, 0], [1, 2], 0.7)
    const draws = gaussianDraws(2000, 1)
    expectInvariant(barkerKernel(t, { stepSize: 0.8 }), draws, { mean: [0, 0], sd: [1, 2] }, { key: key(2), steps: 2 })
    expectInvariant(
      barkerKernel(t, { stepSize: 1, scale: [1, 2] }),
      draws,
      { mean: [0, 0], sd: [1, 2] },
      { key: key(3) },
    )
  })

  it('leaves a skewed target invariant (the Gumbel density: mean 0.5772, sd pi / sqrt 6)', () => {
    const gumbel = numericTarget((x) => -((x[0] ?? 0) + Math.exp(-(x[0] ?? 0))), 1, {
      grad: (x) => [-1 + Math.exp(-(x[0] ?? 0))],
    })
    const u = uniform(key(4), [4000]).data
    const draws = Float64Array.from(u, (v) => -Math.log(-Math.log(Math.min(1 - 1e-12, Math.max(1e-12, v)))))
    expectInvariant(
      barkerKernel(gumbel, { stepSize: 1 }),
      draws,
      { mean: [0.5772156649], sd: [Math.PI / Math.sqrt(6)] },
      { key: key(5), steps: 3 },
    )
  })

  it('samples a Gaussian, and stays stable and moving at a step size that breaks a Langevin proposal', () => {
    const c = sampleChains(barkerKernel(standardNormal(3), { stepSize: 1 }), {
      key: key(6),
      numChains: 2,
      numDraws: 2000,
      numWarmup: 100,
    })
    expectMoments(c, { mean: [0, 0, 0], sd: [1, 1, 1] })
    const wide = sampleChain(barkerKernel(standardNormal(3), { stepSize: 6 }), [0, 0, 0], {
      key: key(7),
      numDraws: 500,
    })
    const a = wide.stats.acceptanceRate ?? new Float64Array()
    expect(a.reduce((x, y) => x + y, 0) / a.length).toBeGreaterThan(0.01)
  })

  it('rejects proposals outside the support and validates its settings', () => {
    const half = numericTarget((x) => ((x[0] ?? 0) > 0 ? -(x[0] ?? 0) : Number.NEGATIVE_INFINITY), 1, {
      grad: () => [-1],
    })
    const r = sampleChain(barkerKernel(half, { stepSize: 0.5 }), [0.5], { key: key(8), numDraws: 200 })
    expect(r.draws.every((v) => v > 0)).toBe(true)
    expect(() => barkerKernel(standardNormal(1), { stepSize: 0 })).toThrow()
    expect(() => barkerKernel(standardNormal(2), { stepSize: 1, scale: [1, -1] })).toThrow()
    expect(() =>
      barkerKernel(
        numericTarget(() => 0, 1),
        { stepSize: 1 },
      ),
    ).toThrow()
  })
})
