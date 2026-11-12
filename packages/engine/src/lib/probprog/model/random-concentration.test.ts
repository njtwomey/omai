// A beta or Dirichlet variable whose concentration is random (a gamma or log-normal hyperprior on the strength of the prior), under
// EP, against the exact posterior by quadrature over the concentration (given it the variable is conjugate).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../message-passing/index.ts'
import { lgammaOf } from '../../message-passing/numerics/_scalar.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toFactorGraph } from './index.ts'

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-10, maxSteps: 300, verdict: true })
  return { c, marginals: marginalsOf(g, r.state as MpState) }
}

/** Exact moments of alpha and of component k of p, for a symmetric Dirichlet(alpha, ..., alpha) and category counts, by quadrature over log alpha. */
function exact(
  counts: readonly number[],
  logPrior: (alpha: number) => number,
  k: number,
): { alpha: number; logAlpha: number; p: number } {
  const K = counts.length
  const N = counts.reduce((p, q) => p + q, 0)
  const T = 6000
  let z = 0
  let ea = 0
  let el = 0
  let ep = 0
  for (let i = 0; i < T; i++) {
    const t = -9 + (i + 0.5) * (18 / T)
    const a = Math.exp(t)
    // log p(counts | alpha) up to a constant: Dirichlet-multinomial.
    const ll = lgammaOf(K * a) - lgammaOf(K * a + N) + counts.reduce((s, c) => s + lgammaOf(a + c) - lgammaOf(a), 0)
    const w = Math.exp(logPrior(a) + t + ll)
    z += w
    ea += w * a
    el += w * t
    ep += (w * (a + (counts[k] as number))) / (K * a + N)
  }
  return { alpha: ea / z, logAlpha: el / z, p: ep / z }
}

describe('a symmetric beta or Dirichlet with a random concentration', () => {
  const gammaPrior = (a: number): number => Math.log(a) - a // Gamma(2, 1) up to a constant
  it('Beta(alpha, alpha) with a gamma alpha: the means of alpha and p within a few percent of the exact posterior', () => {
    const m = defineModel('beta-hyper', (b) => {
      const alpha = b.latent('alpha', d.gamma(2, 1))
      const p = b.latent('p', d.beta(alpha, alpha))
      b.observe('y', d.bernoulli({ probs: p }), b.data('y', { dims: ['i'], support: 'binary' }))
    })
    const y = [1, 0, 1, 1]
    const { marginals, c } = run(m.bind({ y }))
    expect(c.graph?.factors.some((f) => f.operator === 'dirichlet-concentration')).toBe(true)
    const ref = exact([3, 1], gammaPrior, 0)
    const a = marginals['alpha'] as Record<string, number>
    expect(a['mean']).toBeCloseTo(ref.alpha, 1)
    expect(Math.abs((a['mean'] as number) / ref.alpha - 1)).toBeLessThan(0.03)
    expect(Math.abs((marginals['p']?.['mean'] as number) - ref.p)).toBeLessThan(0.01)
  })

  it('a Dirichlet with concentration alpha * ones and a gamma alpha, three categories', () => {
    const counts = [1, 2, 1]
    const m = defineModel('dirichlet-hyper', (b) => {
      const alpha = b.latent('alpha', d.gamma(2, 1))
      const p = b.latent('p', d.dirichlet(ops.mul(alpha, b.const([1, 1, 1]))))
      b.observe('y', d.categorical({ probs: p }), b.data('y', { dims: ['i'], support: 'count' }))
    })
    const { marginals } = run(m.bind({ y: [0, 1, 1, 2] }))
    const ref = exact(counts, gammaPrior, 1)
    expect(Math.abs((marginals['alpha']?.['mean'] as number) / ref.alpha - 1)).toBeLessThan(0.05)
    expect(
      Math.abs((((marginals['p'] as Record<string, unknown>)['mean'] as number[])[1] as number) - ref.p),
    ).toBeLessThan(0.01)
  })

  it('a log-normal concentration (carried on its log) is the numeric rule on a Gaussian', () => {
    const m = defineModel('beta-lognormal', (b) => {
      const alpha = b.latent('alpha', d.logNormal(0.5, 0.8))
      const p = b.latent('p', d.beta(alpha, alpha))
      b.observe('y', d.bernoulli({ probs: p }), b.data('y', { dims: ['i'], support: 'binary' }))
    })
    const { marginals, c } = run(m.bind({ y: [1, 0, 1, 1] }))
    expect(c.graph?.factors.find((f) => f.operator === 'dirichlet-concentration')?.params).toMatchObject({ log: true })
    const logNormalPrior = (a: number): number => (-0.5 * (Math.log(a) - 0.5) ** 2) / 0.64 - Math.log(a)
    const ref = exact([3, 1], logNormalPrior, 0)
    expect(Math.abs((marginals['p']?.['mean'] as number) - ref.p)).toBeLessThan(0.01)
  })

  it('refuses two shapes that are not each exactly one random value (two plain shapes map to beta-two-shapes)', () => {
    const bad = defineModel('bad', (b) => {
      const a1 = b.latent('a1', d.gamma(2, 1))
      const a2 = b.latent('a2', d.gamma(2, 1))
      const p = b.latent('p', d.beta(a1.mul(2), a2))
      b.observe('y', d.bernoulli({ probs: p }), b.data('y', { dims: ['i'], support: 'binary' }))
    })
    const c = toFactorGraph(bad.bind({ y: [1, 0] }))
    expect(c.graph).toBeUndefined()
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/known multiples of one random positive value/)
  })
})
