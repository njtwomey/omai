// A categorical gate is the select factor: EP finds the clusters of a mixture as enumeration does; Dirichlet
// probabilities, known components and their refusals.
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../../message-passing/index.ts'
import { toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from '../index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

describe('a categorical gate is the select factor (expectation propagation)', () => {
  const ys = [-3.1, -2.8, -3.3, 3.0, 2.7, 3.2]
  const mixture = defineModel('mixture', (b) => {
    const n = b.range('row')
    const k = b.range('component')
    const mu = b.latent('mu', d.normal(b.data('m0', { dims: [k] }), 4), { dims: [k] })
    const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [n] })
    b.observe('y', d.normal(ops.at(mu, z), 0.7), b.data('y', { dims: [n] }))
  })
  const bound = mixture.bind({ y: ys, m0: [-1, 1] })

  it('compiles to a categorical variable per row and a select factor per row', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'select').length).toBe(ys.length)
    expect(c.sites.find((s) => s.site === 'z')).toMatchObject({ family: 'discrete', layout: 'elements' })
    expect(c.graph?.variables.find((v) => v.name === 'z[0]')?.family).toEqual({ kind: 'discrete', size: 2 })
  })

  it('EP finds the two cluster means and the assignments of a well-separated mixture, as exact enumeration does', () => {
    const c = toFactorGraph(bound)
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, damping: 0.3 }), { maxSteps: 400 })
    const post = factorGraphPosterior(c, g, r.state as MpState)
    // Exact: enumerate the 2^6 assignments; given one, the cluster means are independent Gaussians.
    const prior = [-1, 1]
    const v0 = 16
    const s2 = 0.49
    let total = 0
    const meanSum = [0, 0]
    const zProb = new Array<number>(ys.length).fill(0)
    // EP settles on one of the two labelings (the prior is weak, so the exact posterior has both): condition on the first row in the first component.
    for (let a = 0; a < 2 ** ys.length; a += 2) {
      const members: number[][] = [[], []]
      for (const [i, y] of ys.entries()) (members[(a >> i) & 1] as number[]).push(y)
      let logw = 0
      const m: number[] = []
      for (const kk of [0, 1]) {
        const ysk = members[kk] as number[]
        const p = 1 / v0 + ysk.length / s2
        const h = (prior[kk] as number) / v0 + ysk.reduce((t, v) => t + v, 0) / s2
        m.push(h / p)
        // log marginal likelihood of the members given the Gaussian prior on the mean
        const sy = ysk.reduce((t, v) => t + v, 0)
        const syy = ysk.reduce((t, v) => t + v * v, 0)
        logw +=
          -0.5 * ysk.length * Math.log(2 * Math.PI * s2) -
          0.5 * Math.log(v0 * p) -
          0.5 * (syy / s2 + (prior[kk] as number) ** 2 / v0 - (h * h) / p) +
          0 * sy
      }
      logw += ys.length * Math.log(0.5)
      const w = Math.exp(logw)
      total += w
      meanSum[0] = (meanSum[0] as number) + w * (m[0] as number)
      meanSum[1] = (meanSum[1] as number) + w * (m[1] as number)
      for (let i = 0; i < ys.length; i++) zProb[i] = (zProb[i] as number) + w * ((a >> i) & 1)
    }
    const mu = toFlat(post.mean('mu'))
    expect(Math.abs((mu[0] as number) - (meanSum[0] as number) / total)).toBeLessThan(0.15)
    expect(Math.abs((mu[1] as number) - (meanSum[1] as number) / total)).toBeLessThan(0.15)
    // Rows 0 to 2 sit at the first mean, rows 3 to 5 at the second.
    const pz = marginalsOf(g, r.state as MpState)
    const probsOf = (i: number): number[] => (pz[`z[${i}]`]?.['probs'] ?? []) as number[]
    for (const i of [0, 1, 2]) expect(probsOf(i)[0] as number).toBeGreaterThan(0.9)
    for (const i of [3, 4, 5]) expect(probsOf(i)[1] as number).toBeGreaterThan(0.9)
    for (const i of [1, 4])
      expect(Math.abs((probsOf(i)[1] as number as number) - (zProb[i] as number) / total)).toBeLessThan(0.05)
  })

  it('a Dirichlet latent as the probabilities is the conjugate discrete factor; a Bernoulli gate in where is a select', () => {
    const weights = defineModel('weights', (b) => {
      const n = b.range('row', { size: 3 })
      const pi = b.latent('pi', d.dirichlet([1, 1]))
      b.latent('z', d.categorical({ probs: pi }), { dims: [n] })
    })
    const cw = toFactorGraph(weights.bind({}))
    expect(cw.graph?.factors.filter((f) => f.operator === 'discrete').length).toBe(3)
    const gate = defineModel('gate', (b) => {
      const n = b.range('row')
      const z = b.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [n] })
      const a = b.latent('a', d.normal(3, 1))
      const c = b.latent('c', d.normal(-3, 1))
      b.observe('y', d.normal(ops.where(z, a, c), 0.5), b.data('y', { dims: [n] }))
    })
    const cg = toFactorGraph(gate.bind({ y: [2.8, -3.2, 3.1] }))
    expect(cg.issues).toEqual([])
    expect(cg.graph?.factors.filter((f) => f.operator === 'select').length).toBe(3)
    const gg = graphOf(cg.graph)
    const r = runAlgorithm(epAlgorithm(gg, { tol: 1e-10, damping: 0.3 }), { maxSteps: 400 })
    const m = marginalsOf(gg, r.state as MpState)
    expect((m['z[0]']?.['p'] as number) > 0.9).toBe(true)
    expect((m['z[1]']?.['p'] as number) < 0.1).toBe(true)
  })

  it('a known component, a computed random index and the normal-gamma idioms are refused with their reasons', () => {
    const known = defineModel('known', (b) => {
      const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }))
      b.observe('y', d.normal(ops.at(b.data('c', { dims: ['k'] }), z), 1), b.data('y', { dims: ['row'] }))
    })
    // Known components are a lookup of the discrete variable (a table over it), not a refusal.
    const lookup = toFactorGraph(known.bind({ c: [0, 1], y: [0.5] }))
    expect(lookup.issues).toEqual([])
    expect(lookup.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'table'])
    const normalGamma = defineModel('ng', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      const mu = b.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(tau, 2)))))
      b.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), b.data('y', { dims: ['row'] }))
    })
    // The pair of a mean and the precision its variance shares, observed with that precision, is one normal-gamma variable.
    const pair = toFactorGraph(normalGamma.bind({ y: [1, 2] }))
    expect(pair.issues).toEqual([])
    expect(pair.graph?.variables.map((v) => v.family.kind)).toContain('normal-gamma')
    // A known multiple of the gamma precision is the scale of the factor (the conditional prior of the mean) when something else
    // reads the mean (here a Gaussian observation with a known scale), so the pair is not one conjugate variable.
    const loose = defineModel('ng-loose', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      const mu = b.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(tau, 2)))))
      b.observe('y', d.normal(mu, 1), b.data('y', { dims: ['row'] }))
    })
    const scaled = toFactorGraph(loose.bind({ y: [1, 2] }))
    expect(scaled.issues).toEqual([])
    expect(scaled.graph?.factors.find((f) => f.id === 'factor:mu')?.params).toEqual({ scale: 2 })
    // The product of two gamma precisions is a gamma variable defined by the `gamma-product` factor (a local times a global precision).
    const product = defineModel('product', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      const kappa = b.latent('kappa', d.gamma(2, 1))
      const mu = b.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(tau, kappa)))))
      b.observe('y', d.normal(mu, 1), b.data('y', { dims: ['row'] }))
    })
    const joint = toFactorGraph(product.bind({ y: [1, 2] }))
    expect(joint.issues).toEqual([])
    expect(joint.graph?.factors.some((f) => f.operator === 'gamma-product')).toBe(true)
    // The square of one gamma variable is not: refused with the idiom named.
    const square = defineModel('square', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      const mu = b.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(tau, tau)))))
      b.observe('y', d.normal(mu, 1), b.data('y', { dims: ['row'] }))
    })
    const issue = toFactorGraph(square.bind({ y: [1, 2] })).issues[0]
    expect(issue?.node).toBe('mu')
    expect(issue?.reason).toMatch(/1 \/ sqrt\(tau\)/)
  })
})
