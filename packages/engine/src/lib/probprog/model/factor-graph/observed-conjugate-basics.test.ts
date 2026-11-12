// Conjugate observations (beta-Bernoulli, gamma-Poisson, beta-binomial, Dirichlet-categorical, a Bernoulli network as
// tables) map to the engine operators and EP is exact.
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
  vmpAlgorithm,
} from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from '../index.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

describe('conjugate observations map to the engine operators and EP is exact', () => {
  const near = (a: number, b: number, tol = 1e-9): void => expect(Math.abs(a - b)).toBeLessThan(tol)

  it('beta-Bernoulli: Beta(a + successes, b + failures), for EP and VMP', () => {
    const coin = defineModel('coin', (m) => {
      m.observe(
        'y',
        d.bernoulli({ probs: m.latent('p', d.beta(2, 3)) }),
        m.data('y', { dims: ['flip'], support: 'binary' }),
      )
    })
    const c = toFactorGraph(coin.bind({ y: [1, 0, 1, 1, 0, 1, 1] }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const ep = factorGraphPosterior(c, g, run(g).state)
    near(Number(toFlat(ep.mean('p'))[0]), (2 + 5) / (2 + 3 + 7))
    near(Number(toFlat(ep.variance('p'))[0]), ((2 + 5) * (3 + 2)) / ((2 + 3 + 7) ** 2 * (2 + 3 + 7 + 1)))
    const vmp = runAlgorithm(vmpAlgorithm(g), { maxSteps: 100 })
    near(Number(toFlat(factorGraphPosterior(c, g, vmp.state as MpState, { rule: 'vmp' }).mean('p'))[0]), 7 / 12)
  })

  it('gamma-Poisson with an exposure, and beta-binomial, one group per element through at', () => {
    const rates = defineModel('rates', (m) => {
      const g = m.range('group')
      const k = m.range('obs')
      const exposure = m.data('exposure', { dims: [k] })
      const code = m.data('code', { dims: [k], support: 'index', indexes: g })
      const lam = m.latent('lam', d.gamma(2, 1), { dims: [g] })
      m.observe('n', d.poisson(ops.mul(exposure, ops.at(lam, code))), m.data('n', { dims: [k], support: 'count' }))
    })
    const c = toFactorGraph(
      rates.bind({ exposure: [1, 2, 1.5, 3], code: [0, 0, 1, 1], n: [3, 5, 2, 7] }, { sizes: { group: 2 } }),
    )
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const post = factorGraphPosterior(c, g, run(g).state)
    // Posterior of a group: Gamma(2 + sum n, 1 + sum exposure).
    const mean = toFlat(post.mean('lam'))
    near(mean[0] as number, (2 + 8) / (1 + 3))
    near(mean[1] as number, (2 + 9) / (1 + 4.5))
    const trials = defineModel('trials', (m) => {
      m.observe('k', d.binomial(10, m.latent('p', d.beta(1, 1))), m.data('k', { dims: ['run'], support: 'count' }))
    })
    const bt = toFactorGraph(trials.bind({ k: [3, 4, 6] }))
    const bg = graphOf(bt.graph)
    near(Number(toFlat(factorGraphPosterior(bt, bg, run(bg).state).mean('p'))[0]), (1 + 13) / (2 + 30))
  })

  it('Dirichlet-categorical: Dirichlet(alpha + counts)', () => {
    const die = defineModel('die', (m) => {
      const pi = m.latent('pi', d.dirichlet([1, 1, 2]))
      m.observe('z', d.categorical({ probs: pi }), m.data('z', { dims: ['roll'], support: 'index' }))
    })
    const c = toFactorGraph(die.bind({ z: [0, 2, 2, 1, 2, 0] }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const mean = toFlat(factorGraphPosterior(c, g, run(g).state).mean('pi'))
    // counts: 2, 1, 3 => alpha = 3, 2, 5, total 10
    for (const [q, v] of [0.3, 0.2, 0.5].entries()) near(mean[q] as number, v)
  })

  it('a Bernoulli network as tables: belief propagation on a tree equals enumeration (explaining away)', () => {
    const net = defineModel('net', (m) => {
      const rain = m.latent('rain', d.bernoulli({ probs: 0.3 }))
      const sprinkler = m.latent('sprinkler', d.bernoulli({ probs: 0.4 }))
      const wetProbs = m.derived(
        'wetProbs',
        ops.where(rain, ops.where(sprinkler, 0.99, 0.8), ops.where(sprinkler, 0.9, 0.05)),
      )
      m.observe('wet', d.bernoulli({ probs: wetProbs }), m.data('wet', { support: 'binary' }))
    })
    const c = toFactorGraph(net.bind({ wet: 1 }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'table'])
    const g = graphOf(c.graph)
    const post = factorGraphPosterior(c, g, run(g).state, { rule: 'bp' })
    // P(rain, sprinkler | wet) by enumeration.
    const joint = (r: number, s: number): number =>
      (r ? 0.3 : 0.7) * (s ? 0.4 : 0.6) * (r ? (s ? 0.99 : 0.8) : s ? 0.9 : 0.05)
    const z = joint(0, 0) + joint(0, 1) + joint(1, 0) + joint(1, 1)
    near(Number(toFlat(post.mean('rain'))[0]), (joint(1, 0) + joint(1, 1)) / z)
    near(Number(toFlat(post.mean('sprinkler'))[0]), (joint(0, 1) + joint(1, 1)) / z)
  })
})

// ----- what is refused -----
