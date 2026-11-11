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
