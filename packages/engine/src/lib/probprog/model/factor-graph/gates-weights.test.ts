// A gate with an unknown mixing weight: a Bernoulli switch whose probability is a beta latent (scalar and plated, with the lookup of known means
// by the switch), a categorical switch with a Dirichlet, and the same mixture with the switch summed out. Against the exact posterior of
// the weight (the switches summed out analytically, a quadrature).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../../message-passing/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, toFactorGraph } from '../index.ts'

const X = [-1.9, 2.1, 0.3, 1.8, -0.2, 0.4, 2.2, -2.1]

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-12, maxSteps: 200 })
  return { c, marginals: marginalsOf(g, r.state as MpState) }
}

const density = (x: number, m: number): number => Math.exp(-0.5 * ((x - m) / 0.5) ** 2) / (0.5 * Math.sqrt(2 * Math.PI))

/** The exact mean and variance of the weight with a Beta(a, b) prior, by the midpoint rule. */
function exactWeight(a: number, b: number): { mean: number; variance: number } {
  const T = 20000
  let z = 0
  let m1 = 0
  let m2 = 0
  for (let i = 0; i < T; i++) {
    const p = (i + 0.5) / T
    const w = Math.exp(
      (a - 1) * Math.log(p) +
        (b - 1) * Math.log(1 - p) +
        X.reduce((s, x) => s + Math.log(p * density(x, 2) + (1 - p) * density(x, -2)), 0),
    )
    z += w
    m1 += w * p
    m2 += w * p * p
  }
  return { mean: m1 / z, variance: m2 / z - (m1 / z) ** 2 }
}

describe('a gate with an unknown mixing weight', () => {
  const [a, b] = [1.5, 2]
  const exact = exactWeight(a, b)
  const explicit = defineModel('gate-w', (m) => {
    const i = m.range('i')
    const pi = m.latent('pi', d.beta(a, b))
    const z = m.latent('z', d.bernoulli({ probs: pi }), { dims: [i] })
    const x = m.data('x', { dims: [i] })
    m.observe('x', d.normal(ops.sub(ops.mul(z, 4), 2), 0.5), x)
  })
  const summed = defineModel('gate-w-summed', (m) => {
    const i = m.range('i')
    const pi = m.latent('pi', d.beta(a, b))
    const x = m.data('x', { dims: [i] })
    m.observe('x', d.normalPair(pi, -2, 2, 0.5), x)
  })

  it('the explicit switches (a Bernoulli over a beta) and the summed-out mixture give the same beta for the weight, close to the exact posterior', () => {
    const e = run(explicit.bind({ x: X }))
    const s = run(summed.bind({ x: X }))
    expect(e.c.graph?.factors.filter((f) => f.operator === 'bernoulli').length).toBe(X.length)
    expect(s.c.graph?.factors.filter((f) => f.operator === 'mixture-weight').length).toBe(X.length)
    const pe = e.marginals['pi'] as Record<string, number>
    const ps = s.marginals['pi'] as Record<string, number>
    expect(pe['a']).toBeCloseTo(ps['a'] as number, 6)
    expect(pe['b']).toBeCloseTo(ps['b'] as number, 6)
    expect(pe['mean']).toBeCloseTo(exact.mean, 2)
    expect(Math.sqrt(pe['variance'] as number)).toBeCloseTo(Math.sqrt(exact.variance), 2)
  })

  it('a Bernoulli over a beta that is not a beta (a known value) or has no beta is refused with its reason', () => {
    const bad = defineModel('bad', (m) => {
      const g = m.latent('g', d.gamma(2, 1))
      const z = m.latent('z', d.bernoulli({ probs: g }))
      const x = m.data('x', { dims: ['i'] })
      m.observe('x', d.normal(ops.mul(z, 4), 0.5), x)
    })
    const c = toFactorGraph(bad.bind({ x: X }))
    expect(c.graph).toBeUndefined()
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/beta latent/)
  })

  it('a categorical switch with a Dirichlet over three known means', () => {
    const three = defineModel('gate-w3', (m) => {
      const i = m.range('i')
      const pi = m.latent('pi', d.dirichlet([1, 1, 1]))
      const z = m.latent('z', d.categorical({ probs: pi }), { dims: [i] })
      const means = m.data('means', { dims: ['k'] })
      const x = m.data('x', { dims: [i] })
      m.observe('x', d.normal(ops.at(means, z), 0.5), x)
    })
    const { marginals } = run(three.bind({ x: X, means: [-2, 0, 2] }))
    const mean = (marginals['pi'] as Record<string, unknown>)['mean'] as number[]
    // Three observations near each end and two in the middle: the weights follow the counts.
    expect(mean[0]).toBeGreaterThan(0.2)
    expect(mean.reduce((p, q) => p + q, 0)).toBeCloseTo(1, 10)
  })
})
