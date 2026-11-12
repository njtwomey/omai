// The conjugate observations added with the prior atlas: an exponential or gamma value with a gamma rate, a geometric or negative
// binomial count with a beta probability. EP is exact (the posterior's parameters and the log evidence against a quadrature).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { evidenceOf, type MpState, marginalsOf, projectionOf } from '../../../message-passing/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, type toFactorGraph } from '../index.ts'

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-12, maxSteps: 100 })
  return {
    c,
    marginals: marginalsOf(g, r.state as MpState),
    evidence: evidenceOf(g, { projection: projectionOf('ep') }, r.state as MpState),
  }
}

/** Trapezoid integral of f over (lo, hi) on a fine grid. */
function integrate(f: (x: number) => number, lo: number, hi: number, n = 200000): number {
  const h = (hi - lo) / n
  let s = 0.5 * (f(lo) + f(hi))
  for (let i = 1; i < n; i++) s += f(lo + i * h)
  return s * h
}
const lgamma = (x: number): number => {
  let s = 0
  let z = x
  while (z < 10) {
    s -= Math.log(z)
    z += 1
  }
  return (
    s +
    (z - 0.5) * Math.log(z) -
    z +
    0.5 * Math.log(2 * Math.PI) +
    1 / (12 * z) -
    1 / (360 * z ** 3) +
    1 / (1260 * z ** 5)
  )
}

describe('an exponential or gamma observation with a gamma rate', () => {
  const Y = [0.8, 1.9, 0.3, 1.2]
  const [a, b] = [2, 1.5]
  it('the exponential rate is the gamma posterior, with the log evidence of the quadrature', () => {
    const m = defineModel('exp', (bld) => {
      const r = bld.latent('r', d.gamma(a, b))
      bld.observe('y', d.exponential(r), bld.data('y', { dims: ['i'] }))
    })
    const { marginals, evidence } = run(m.bind({ y: Y }))
    const sum = Y.reduce((p, q) => p + q, 0)
    const r = marginals['r'] as Record<string, number>
    expect(r['shape']).toBeCloseTo(a + Y.length, 10)
    expect(r['rate']).toBeCloseTo(b + sum, 10)
    const logPrior = (x: number): number => a * Math.log(b) - lgamma(a) + (a - 1) * Math.log(x) - b * x
    const z = integrate((x) => Math.exp(logPrior(x) + Y.length * Math.log(x) - x * sum), 1e-9, 40)
    expect(evidence).toBeCloseTo(Math.log(z), 5)
  })
  it('a gamma value with a known shape and a known multiple of the rate', () => {
    const m = defineModel('gam', (bld) => {
      const r = bld.latent('r', d.gamma(a, b))
      bld.observe('y', d.gamma(3, ops.mul(2, r)), bld.data('y', { dims: ['i'] }))
    })
    const { marginals, evidence } = run(m.bind({ y: Y }))
    const sum = Y.reduce((p, q) => p + q, 0)
    const r = marginals['r'] as Record<string, number>
    expect(r['shape']).toBeCloseTo(a + 3 * Y.length, 10)
    expect(r['rate']).toBeCloseTo(b + 2 * sum, 10)
    const logPrior = (x: number): number => a * Math.log(b) - lgamma(a) + (a - 1) * Math.log(x) - b * x
    const logLik = (x: number): number =>
      Y.reduce((p, y) => p + 3 * Math.log(2 * x) + 2 * Math.log(y) - lgamma(3) - 2 * x * y, 0)
    const z = integrate((x) => Math.exp(logPrior(x) + logLik(x)), 1e-9, 40)
    expect(evidence).toBeCloseTo(Math.log(z), 5)
  })
})

describe('a geometric or negative binomial count with a beta probability', () => {
  const [a, b] = [2, 3]
  it('geometric: the beta posterior and its evidence', () => {
    const Y = [3, 5, 2, 4]
    const m = defineModel('geo', (bld) => {
      const p = bld.latent('p', d.beta(a, b))
      bld.observe('y', d.geometric(p), bld.data('y', { dims: ['i'], support: 'count' }))
    })
    const { marginals, evidence } = run(m.bind({ y: Y }))
    const t = marginals['p'] as Record<string, number>
    expect(t['a']).toBeCloseTo(a + Y.length, 10)
    expect(t['b']).toBeCloseTo(b + Y.reduce((s, y) => s + y - 1, 0), 10)
    const logPrior = (x: number): number =>
      lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)
    const logLik = (x: number): number => Y.reduce((s, y) => s + Math.log(x) + (y - 1) * Math.log(1 - x), 0)
    expect(evidence).toBeCloseTo(Math.log(integrate((x) => Math.exp(logPrior(x) + logLik(x)), 1e-12, 1 - 1e-12)), 5)
  })
  it('negative binomial with a known number of successes', () => {
    const Y = [3, 0, 2, 5]
    const r = 3
    const m = defineModel('nb', (bld) => {
      const p = bld.latent('p', d.beta(a, b))
      bld.observe('y', d.negativeBinomial(r, p), bld.data('y', { dims: ['i'], support: 'count' }))
    })
    const { marginals, evidence } = run(m.bind({ y: Y }))
    const t = marginals['p'] as Record<string, number>
    expect(t['a']).toBeCloseTo(a + r * Y.length, 10)
    expect(t['b']).toBeCloseTo(b + Y.reduce((s, y) => s + y, 0), 10)
    const logPrior = (x: number): number =>
      lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)
    const logLik = (x: number): number =>
      Y.reduce((s, y) => s + lgamma(y + r) - lgamma(y + 1) - lgamma(r) + r * Math.log(x) + y * Math.log(1 - x), 0)
    expect(evidence).toBeCloseTo(Math.log(integrate((x) => Math.exp(logPrior(x) + logLik(x)), 1e-12, 1 - 1e-12)), 5)
  })
})
