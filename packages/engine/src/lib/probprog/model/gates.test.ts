// Gates in the factor-graph compiler: a discrete latent that chooses a known mean or a known transition row, against the exact
// answers of the benchmark models (forward-backward for the Gaussian HMM, the closed-form mixture for the bimodal target).
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
  vmpAlgorithm,
} from '../../message-passing/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { bimodal, gaussianHmm } from '../models/index.ts'
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
})
