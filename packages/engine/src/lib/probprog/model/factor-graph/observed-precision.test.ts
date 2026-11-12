// A random precision is the gaussian-gamma-precision factor: the mean-field fixed point, the hierarchy, and EP
// against the exact posteriors.
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
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph } from '../index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

const _N = 40

describe('a random precision is the gaussian-gamma-precision factor (variational message passing)', () => {
  const ys = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]
  const model = defineModel('precision', (b) => {
    const tau = b.latent('tau', d.gamma(2, 1))
    const mu = b.latent('mu', d.normal(0, 10))
    b.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), b.data('y', { dims: ['row'] }))
  })

  it('compiles and the fixed point is the closed-form mean field of the conjugate model', () => {
    const c = toFactorGraph(model.bind({ y: ys }))
    expect(c.issues).toEqual([])
    expect(new Set(c.graph?.factors.map((f) => f.operator))).toEqual(new Set(['prior', 'gaussian-gamma-precision']))
    const g = graphOf(c.graph)
    const r = runAlgorithm(vmpAlgorithm(g, { tol: 1e-13 }), { maxSteps: 500 })
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'vmp' })
    // Closed form (Bishop 10.1.3 with an independent prior on mu): q(mu) = N(m, 1 / lambda), q(tau) = Gamma(a, b).
    const n = ys.length
    const sum = ys.reduce((t, v) => t + v, 0)
    const a = 2 + n / 2
    let bb = 1
    let m = 0
    let lambda = 1 / 100
    for (let k = 0; k < 500; k++) {
      const e = a / bb
      lambda = 1 / 100 + n * e
      m = (e * sum) / lambda
      bb = 1 + 0.5 * ys.reduce((t, v) => t + (v - m) ** 2 + 1 / lambda, 0)
    }
    expect(toFlat(post.mean('mu'))[0]).toBeCloseTo(m, 8)
    expect(toFlat(post.stddev('mu'))[0]).toBeCloseTo(Math.sqrt(1 / lambda), 8)
    expect(toFlat(post.mean('tau'))[0]).toBeCloseTo(a / bb, 8)
    // The variational bound rises at every sweep.
    const alg = vmpAlgorithm(g, { tol: 1e-13 })
    let st = alg.init(undefined as never)
    let last = Number.NEGATIVE_INFINITY
    for (let k = 0; k < 15; k++) {
      st = alg.step(st, undefined as never)
      const bound = factorGraphPosterior(c, g, st, { rule: 'vmp' }).logEvidence?.value
      if (bound !== undefined) {
        expect(bound).toBeGreaterThanOrEqual(last - 1e-9)
        last = bound
      }
    }
    expect(last).toBeGreaterThan(Number.NEGATIVE_INFINITY)
  })

  it('maps pow(tau, -0.5) and a hierarchy with a random precision on the group means', () => {
    const hier = defineModel('hier', (b) => {
      const j = b.range('group')
      const tau = b.latent('tau', d.exponential(1))
      const theta = b.latent('theta', d.normal(0, ops.pow(tau, -0.5)), { dims: [j] })
      b.observe('y', d.normal(theta, 1), b.data('y', { dims: [j] }))
    })
    const c = toFactorGraph(hier.bind({ y: [1, -1, 2] }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const r = runAlgorithm(vmpAlgorithm(g), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    expect(toFlat(factorGraphPosterior(c, g, r.state as MpState, { rule: 'vmp' }).mean('theta')).length).toBe(3)
  })

  it('runs under expectation propagation (moments by quadrature in log lambda) close to the exact two-dimensional posterior', () => {
    const c = toFactorGraph(model.bind({ y: ys }))
    const g = graphOf(c.graph)
    expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 400 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    // The exact posterior of (mu, tau) on a grid: mu ~ N(0, 10^2), tau ~ Gamma(2, 1), y_i ~ N(mu, 1 / tau).
    let z = 0
    let eMu = 0
    let eMu2 = 0
    let eTau = 0
    for (let i = 0; i < 400; i++) {
      const mu = -4 + (8 * (i + 0.5)) / 400
      for (let j = 0; j < 400; j++) {
        const tau = (12 * (j + 0.5)) / 400
        const logp =
          -0.5 * (mu / 10) ** 2 +
          Math.log(tau) -
          tau +
          ys.reduce((t, v) => t + 0.5 * Math.log(tau) - 0.5 * tau * (v - mu) ** 2, 0)
        const w = Math.exp(logp)
        z += w
        eMu += w * mu
        eMu2 += w * mu * mu
        eTau += w * tau
      }
    }
    const mean = eMu / z
    const sd = Math.sqrt(eMu2 / z - mean * mean)
    expect(toFlat(post.mean('mu'))[0]).toBeCloseTo(mean, 1)
    expect(toFlat(post.stddev('mu'))[0] ?? 0).toBeGreaterThan(0.8 * sd)
    expect(toFlat(post.stddev('mu'))[0] ?? 0).toBeLessThan(1.25 * sd)
    expect(Math.abs((toFlat(post.mean('tau'))[0] ?? 0) / (eTau / z) - 1)).toBeLessThan(0.1)
  })

  it('a linear regression with an unknown noise level: expectation propagation finds the coefficients the exact posterior does', () => {
    const X = [
      [1, -1],
      [1, -0.4],
      [1, 0.1],
      [1, 0.8],
      [1, 1.5],
      [1, 0.3],
    ]
    const y = [-1.1, -0.2, 0.3, 1.2, 2.1, 0.6]
    const regression = defineModel('lr', (b) => {
      const n = b.range('row')
      const f = b.range('feature')
      const Xd = b.data('X', { dims: [n, f] })
      const w = b.latent('w', d.normal(0, 2), { dims: [f] })
      const tau = b.latent('tau', d.gamma(2, 1))
      b.observe('y', d.normal(Xd.matmul(w), ops.div(1, ops.sqrt(tau))), b.data('y', { dims: [n] }))
    })
    const c = toFactorGraph(regression.bind({ X, y }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 600 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    // Exact: given tau the coefficients are Gaussian (precision tau X'X + I / 4, mean tau P^-1 X'y); tau has its own marginal posterior.
    const sxx = [0, 0, 0, 0]
    const sxy = [0, 0]
    let syy = 0
    for (const [i, row] of X.entries()) {
      const yi = y[i] as number
      for (let a = 0; a < 2; a++) {
        sxy[a] = (sxy[a] as number) + (row[a] as number) * yi
        for (let b2 = 0; b2 < 2; b2++)
          sxx[a * 2 + b2] = (sxx[a * 2 + b2] as number) + (row[a] as number) * (row[b2] as number)
      }
      syy += yi * yi
    }
    let z = 0
    const eW = [0, 0]
    let eTau = 0
    for (let j = 0; j < 4000; j++) {
      const tau = (20 * (j + 0.5)) / 4000
      const p00 = tau * (sxx[0] as number) + 0.25
      const p01 = tau * (sxx[1] as number)
      const p11 = tau * (sxx[3] as number) + 0.25
      const det = p00 * p11 - p01 * p01
      const b0 = tau * (sxy[0] as number)
      const b1 = tau * (sxy[1] as number)
      const m0 = (p11 * b0 - p01 * b1) / det
      const m1 = (p00 * b1 - p01 * b0) / det
      const logp =
        Math.log(tau) -
        tau +
        (X.length / 2) * Math.log(tau) -
        0.5 * Math.log(det) -
        0.5 * tau * syy +
        0.5 * (b0 * m0 + b1 * m1)
      const wt = Math.exp(logp)
      z += wt
      eW[0] = (eW[0] as number) + wt * m0
      eW[1] = (eW[1] as number) + wt * m1
      eTau += wt * tau
    }
    const mean = toFlat(post.mean('w'))
    expect(mean[0]).toBeCloseTo((eW[0] as number) / z, 1)
    expect(mean[1]).toBeCloseTo((eW[1] as number) / z, 1)
    expect(Math.abs((toFlat(post.mean('tau'))[0] ?? 0) / (eTau / z) - 1)).toBeLessThan(0.15)
  })

  it('a population of latent values with an unknown spread runs under expectation propagation and finds the exact posterior', () => {
    const y = [1.4, -0.6, 2.1, 0.3, -1.2, 0.9]
    const hier = defineModel('population', (b) => {
      const j = b.range('group')
      const tau = b.latent('tau', d.gamma(2, 1))
      const theta = b.latent('theta', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [j] })
      b.observe('y', d.normal(theta, 1), b.data('y', { dims: [j] }))
    })
    const c = toFactorGraph(hier.bind({ y }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 600 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    // Exact: y_j | tau ~ N(0, 1 + 1 / tau), so tau has a one-dimensional posterior; given tau, theta_j has mean y_j / (1 + tau).
    let z = 0
    let eTau = 0
    const eTheta = y.map(() => 0)
    for (let i = 0; i < 6000; i++) {
      const tau = (30 * (i + 0.5)) / 6000
      const v = 1 + 1 / tau
      const logp = Math.log(tau) - tau - 0.5 * y.reduce((t, x) => t + Math.log(v) + (x * x) / v, 0)
      const w = Math.exp(logp)
      z += w
      eTau += w * tau
      for (const [k, x] of y.entries()) eTheta[k] = (eTheta[k] as number) + (w * x) / (1 + tau)
    }
    const means = toFlat(post.mean('theta'))
    for (const [k, e] of eTheta.entries()) expect(means[k]).toBeCloseTo(e / z, 1)
    expect(Math.abs((toFlat(post.mean('tau'))[0] ?? 0) / (eTau / z) - 1)).toBeLessThan(0.2)
  })

  it('a scale that is not the idiom is refused with it named', () => {
    const wrong = defineModel('wrong', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      b.observe('y', d.normal(0, ops.exp(tau)), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(wrong.bind({ y: [1] })).issues[0]?.reason).toMatch(/1 \/ sqrt\(tau\)/)
    const notGamma = defineModel('notgamma', (b) => {
      const s = b.latent('s', d.normal(1, 1))
      b.observe('y', d.normal(0, ops.div(1, ops.sqrt(s))), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(notGamma.bind({ y: [1] })).issues[0]?.reason).toMatch(/must be a gamma or exponential latent/)
  })
})
