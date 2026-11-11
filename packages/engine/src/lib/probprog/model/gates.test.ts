// Gates in the factor-graph compiler: a discrete latent that chooses a known mean or a known transition row, against the exact
// answers of the benchmark models (forward-backward for the Gaussian HMM, the closed-form mixture for the bimodal target).
import { describe, expect, it } from 'vitest'
import {
  clutterGraph,
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
  vmpAlgorithm,
} from '../../message-passing/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { bimodal, clutterProblem, gaussianHmm } from '../models/index.ts'
import { zeroInflatedPoisson } from '../validation/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphVerdict, ops, toFactorGraph } from './index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0], options = {}) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, ...options }), { maxSteps: 400 })
  return { c, g, r, marginals: marginalsOf(g, r.state as MpState) }
}

describe('a discrete state that chooses a known mean or a known transition row', () => {
  it('the Gaussian HMM is a chain of tables and EP is exact: the state marginals match forward-backward', () => {
    const bound = gaussianHmm.bound()
    const truth = gaussianHmm.truth()
    const { c, marginals } = ep(bound)
    expect(c.graph?.factors.some((f) => f.operator === 'table')).toBe(true)
    for (let t = 0; t < 16; t++) {
      const probs = (marginals[`z[${t}]`] as { probs: number[] }).probs
      const mean = probs.reduce((a, p, k) => a + p * k, 0)
      expect(mean).toBeCloseTo(truth.marginals[`z[${t}]`]?.mean as number, 6)
    }
  })

  it('the bimodal target is a gate over a Gaussian per state, and the moments of the marginals are the mixture moments', () => {
    const { c, marginals } = ep(bimodal.bound())
    expect(c.graph?.factors.filter((f) => f.operator === 'gate').length).toBe(2)
    const truth = bimodal.truth()
    for (const i of [0, 1]) {
      const m = marginals[`x[${i}]`] as { mean: number; variance: number }
      expect(m.mean).toBeCloseTo(truth.marginals[`x[${i}]`]?.mean as number, 1)
      expect(Math.sqrt(m.variance)).toBeCloseTo(truth.marginals[`x[${i}]`]?.sd as number, 1)
    }
  })

  it('a latent mean with a known value per state and a Gaussian observation: the state posterior is the Bayes factor', () => {
    const m = defineModel('known-means', (b) => {
      const z = b.latent('z', d.categorical({ probs: [0.3, 0.7] }))
      const x = b.latent('x', d.normal(ops.at(b.data('c', { dims: ['k'] }), z), 0.5))
      b.observe('y', d.normal(x, 0.5), b.data('y', { dims: ['row'] }))
    })
    const y = 1.3
    const { marginals } = ep(m.bind({ c: [0, 2], y: [y] }))
    // y given the state is N(c_z, 0.25 + 0.25): the posterior of z by Bayes.
    const lik = (c: number): number => Math.exp(-((y - c) ** 2) / (2 * 0.5)) / Math.sqrt(2 * Math.PI * 0.5)
    const p1 = (0.7 * lik(2)) / (0.3 * lik(0) + 0.7 * lik(2))
    const got = (marginals['z'] as { probs: number[] }).probs
    expect(got[1]).toBeCloseTo(p1, 6)
    // The posterior mean of x: the mixture of the two Gaussian posteriors (precision 8 each, means (c 4 + y 4) / 8).
    const mean = (1 - p1) * ((0 * 4 + y * 4) / 8) + p1 * ((2 * 4 + y * 4) / 8)
    expect((marginals['x'] as { mean: number }).mean).toBeCloseTo(mean, 6)
  })

  it('the variational rule runs on the table and gate forms too (a gate runs with the expected log of its branches)', () => {
    const c = toFactorGraph(bimodal.bound())
    const g = compileGraph(c.graph as never, factorOperators())
    expect(factorGraphVerdict(c, g, 'vmp').ok).toBe(true)
    const r = runAlgorithm(vmpAlgorithm(g, { tol: 1e-8 }), { maxSteps: 200 })
    expect(['converged', 'completed']).toContain(r.status)
  })

  it('the clutter problem is a gate per observation over the location and a known clutter component, and EP is near the quadrature truth', () => {
    const bound = clutterProblem.bound()
    const truth = clutterProblem.truth()
    const { c, marginals } = ep(bound, { damping: 0.5 })
    const rows = bound.data['y']?.shape[0] as number
    expect(c.graph?.factors.filter((f) => f.operator === 'gate').length).toBe(rows)
    const m = marginals['theta'] as { mean: number; variance: number }
    // The same EP problem written as one numeric clutter factor per point (`clutterGraph`): the gate and the numeric rule must agree.
    const ys = Array.from(toFlat(bound.data['y'] as never))
    const g = clutterGraph({
      ys,
      truth: 2,
      weight: 0.5,
      clutterVariance: 10,
      priorVariance: 100,
      isClutter: ys.map(() => false),
    })
    const reference = runAlgorithm(epAlgorithm(g, { tol: 1e-9, damping: 0.5 }), { maxSteps: 400 })
    const ref = marginalsOf(g, reference.state as MpState)['x'] as { mean: number; variance: number }
    expect(m.mean).toBeCloseTo(ref.mean, 1)
    expect(Math.sqrt(m.variance) / Math.sqrt(ref.variance)).toBeCloseTo(1, 1)
    // The exact posterior is bimodal (a mode at the location and one at a distractor cluster); EP, as always, settles on one of them.
    const oracle = truth.marginals['theta']
    expect(m.mean).toBeGreaterThan((oracle?.mean as number) - 2 * (oracle?.sd as number))
    expect(m.mean).toBeLessThan((oracle?.mean as number) + 2 * (oracle?.sd as number))
  })

  it('a change point: the segment of each row is a known function of a discrete change time, a gate over the two random means', () => {
    const K = 8
    const ts = Array.from({ length: 10 }, (_, t) => t)
    const ys = [0.1, -0.3, 0.2, 0.0, -0.1, 0.4, 2.1, 1.8, 2.3, 1.9]
    const model = defineModel('change-point', (b) => {
      const row = b.range('row')
      const seg = b.range('segment', { size: 2 })
      const tau = b.latent('tau', d.categorical({ probs: new Array<number>(K).fill(1 / K) }))
      const mu = b.latent('mu', d.normal(0, 3), { dims: [seg] })
      const t = b.data('t', { dims: [row] })
      b.observe('y', d.normal(ops.at(mu, ops.gt(t, ops.add(tau, 0))), 0.5), b.data('y', { dims: [row] }))
    })
    const { c, marginals } = ep(model.bind({ t: ts, y: ys }), { damping: 0.4 })
    expect(c.graph?.factors.filter((f) => f.operator === 'gate').length).toBe(ys.length)
    // The exact posterior: for each change time k (rows t > k use the second mean) the two means are conjugate Gaussians.
    const s2 = 0.25
    const logMarginal = (v: number[]): number => {
      const n = v.length
      if (n === 0) return 0
      const sum = v.reduce((a, b) => a + b, 0)
      const ss = v.reduce((a, b) => a + b * b, 0)
      const prec = 1 / 9 + n / s2
      return (
        -0.5 * n * Math.log(2 * Math.PI * s2) -
        0.5 * Math.log(9) -
        0.5 * Math.log(prec) -
        ss / (2 * s2) +
        (sum / s2) ** 2 / (2 * prec)
      )
    }
    const logs = Array.from(
      { length: K },
      (_, k) => logMarginal(ys.filter((_, t) => !(t > k))) + logMarginal(ys.filter((_, t) => t > k)),
    )
    const top = Math.max(...logs)
    const w = logs.map((v) => Math.exp(v - top))
    const total = w.reduce((a, b) => a + b, 0)
    const exact = w.map((v) => v / total)
    const got = (marginals['tau'] as { probs: number[] }).probs
    const mode = exact.indexOf(Math.max(...exact))
    expect(got.indexOf(Math.max(...got))).toBe(mode)
    for (let k = 0; k < K; k++) expect(Math.abs((got[k] as number) - (exact[k] as number))).toBeLessThan(0.12)
    const m1 = marginals['mu[1]'] as { mean: number }
    expect(m1.mean).toBeGreaterThan(1.5)
  })
})

describe('a gate in the mean with a random noise precision', () => {
  it('a spike and slab switch per coefficient and a gamma precision compile, and EP finds the active coefficient and the noise', () => {
    const X = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2].map((v, i) => [v, Math.cos(2 * i)])
    const y = X.map((r, i) => 2 * (r[0] as number) + 0.1 * Math.sin(5 * i))
    const m = defineModel('gate-precision', (b) => {
      const row = b.range('row')
      const feature = b.range('feature')
      const z = b.latent('z', d.bernoulli({ probs: 0.3 }), { dims: [feature] })
      const slab = b.latent('slab', d.normal(0, 3), { dims: [feature] })
      const spike = b.latent('spike', d.normal(0, 0.05), { dims: [feature] })
      const tau = b.latent('tau', d.gamma(2, 0.1))
      const mean = b.data('X', { dims: [row, feature] }).matmul(ops.where(z, slab, spike))
      b.observe('y', d.normal(mean, ops.div(1, ops.sqrt(tau))), b.data('y', { dims: [row] }))
    })
    const { marginals } = ep(m.bind({ X, y }), { damping: 0.5 })
    const p = (name: string): number => (marginals[name] as unknown as { p: number }).p
    expect(p('z[0]')).toBeGreaterThan(0.95)
    expect(p('z[1]')).toBeLessThan(0.5)
    expect((marginals['tau'] as { mean: number }).mean).toBeGreaterThan(20)
  })
})

describe('a gate over a Poisson likelihood: zero inflation', () => {
  const x = [-1, -0.5, 0, 0.5, 1, 1.2, -1.2, 0.3, 0.1, -0.3, 0.7, 0.9]
  const y = [0, 1, 0, 2, 4, 0, 0, 1, 0, 0, 3, 2]
  const zip = defineModel('zero-inflated', (b) => {
    const row = b.range('row')
    const state = b.range('state')
    const w = b.latent('w', d.dirichlet([1, 1]))
    const z = b.latent('z', d.categorical({ probs: w }), { dims: [row] })
    const eta = ops.add(
      b.latent('b0', d.normal(0, 2)),
      ops.mul(b.latent('b1', d.normal(0, 2)), b.data('x', { dims: [row] })),
    )
    // The rate is the exposure the state gives (1: a Poisson count, 0: a structural zero) times exp(eta).
    b.observe(
      'y',
      d.poisson(ops.mul(ops.at(b.data('on', { dims: [state] }), z), ops.exp(eta))),
      b.data('y', { dims: [row] }),
    )
  })

  it('compiles to a gate per row and a table that rules out a structural zero for each positive count', () => {
    const c = toFactorGraph(zip.bind({ x, y, on: [1, 0] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'gate')).toHaveLength(y.length)
    expect(c.graph?.factors.filter((f) => f.operator === 'table')).toHaveLength(y.filter((v) => v > 0).length)
  })

  it('EP agrees with the exact posterior (the structural zero marginalised, on a grid over the inflation and both coefficients)', () => {
    const { marginals } = ep(zip.bind({ x, y, on: [1, 0] }), { damping: 0.5 })
    const post = zeroInflatedPoisson({ x, y, inflation: { a: 1, b: 1 }, coefficientSd: 2 })
    const exact = [post.pi, post.b0, post.b1]
    const got = [
      (marginals['w'] as unknown as { mean: number[] }).mean[1] as number,
      (marginals['b0'] as { mean: number }).mean,
      (marginals['b1'] as { mean: number }).mean,
    ]
    for (const k of [0, 1, 2])
      expect(Math.abs((got[k] as number) - (exact[k]?.mean as number))).toBeLessThan(0.2 * (exact[k]?.sd as number))
    // A positive count cannot be a structural zero.
    expect((marginals['z[1]'] as unknown as { probs: number[] }).probs[1]).toBeCloseTo(0, 6)
  })
})
