// The normal-inverse-gamma prior on a Gaussian's mean and variance under EP: an inverse-gamma variance is a gamma precision, and the
// mean with the variance `sigma2 / kappa` shares it, so the pair is one normal-gamma variable and EP is exact (against the closed form).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { evidenceOf, type MpState, marginalsOf, projectionOf } from '../../../message-passing/index.ts'
import { item } from '../../../tensor/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, ops, toFactorGraph } from '../index.ts'

const Y = [1.2, 0.4, 1.9, 1.1, 0.7, 1.5]
const lg = (x: number): number => {
  // Stirling with shift: accurate to 1e-12 for the arguments used here.
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

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-12, maxSteps: 100, verdict: true })
  return { c, g, r, marginals: marginalsOf(g, r.state as MpState) }
}

describe('the normal-inverse-gamma prior on a mean and a variance', () => {
  const [m0, kappa, a, b] = [0.2, 0.5, 3, 2]
  const model = defineModel('nig', (bld) => {
    const sigma2 = bld.latent('sigma2', d.inverseGamma(a, b))
    const mu = bld.latent('mu', d.normal(m0, ops.sqrt(ops.div(sigma2, kappa))))
    bld.observe('y', d.normal(mu, ops.sqrt(sigma2)), bld.data('y', { dims: ['i'], role: 'response' }))
  })
  const n = Y.length
  const ybar = Y.reduce((p, q) => p + q, 0) / n
  const S = Y.reduce((p, q) => p + (q - ybar) ** 2, 0)
  const kn = kappa + n
  const mn = (kappa * m0 + n * ybar) / kn
  const an = a + n / 2
  const bn = b + S / 2 + (kappa * n * (ybar - m0) ** 2) / (2 * kn)

  it('compiles to one normal-gamma variable with a prior and a factor per observation, and EP is the closed-form posterior', () => {
    const { c, g, r, marginals } = run(model.bind({ y: Y }))
    expect(c.graph?.variables.filter((v) => v.family.kind === 'normal-gamma').length).toBe(1)
    expect(c.graph?.factors.filter((f) => f.operator === 'gaussian-normal-gamma').length).toBe(n)
    const t = marginals['mu'] as Record<string, number>
    expect(t['m']).toBeCloseTo(mn, 10)
    expect(t['beta']).toBeCloseTo(kn, 10)
    expect(t['a']).toBeCloseTo(an, 10)
    expect(t['b']).toBeCloseTo(bn, 10)
    const post = factorGraphPosterior(c, g, r.state as MpState)
    // The marginal of the mean is a Student t, of the variance an inverse gamma: their moments are exact too.
    expect(item(post.mean('mu'))).toBeCloseTo(mn, 10)
    expect(item(post.stddev('mu'))).toBeCloseTo(Math.sqrt(bn / ((an - 1) * kn)), 8)
    expect(item(post.mean('sigma2'))).toBeCloseTo(bn / (an - 1), 10)
    expect(item(post.stddev('sigma2'))).toBeCloseTo(Math.sqrt(bn ** 2 / ((an - 1) ** 2 * (an - 2))), 8)
  })

  it('has the log evidence of the closed form', () => {
    const { g, r } = run(model.bind({ y: Y }))
    const logZ =
      lg(an) -
      lg(a) +
      a * Math.log(b) -
      an * Math.log(bn) +
      0.5 * Math.log(kappa / kn) -
      (n / 2) * Math.log(2 * Math.PI)
    expect(evidenceOf(g, { projection: projectionOf('ep') }, r.state as MpState)).toBeCloseTo(logZ, 8)
  })

  it('a gamma precision written the precision way is the same pair', () => {
    const tauModel = defineModel('ng', (bld) => {
      const tau = bld.latent('tau', d.gamma(a, b))
      const mu = bld.latent('mu', d.normal(m0, ops.div(1, ops.sqrt(ops.mul(kappa, tau)))))
      bld.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), bld.data('y', { dims: ['i'], role: 'response' }))
    })
    const { marginals } = run(tauModel.bind({ y: Y }))
    expect((marginals['mu'] as Record<string, number>)['b']).toBeCloseTo(bn, 10)
  })

  it('observations with a known precision multiple keep the pair exact', () => {
    const w = [2, 2, 2, 2, 2, 2]
    const weighted = defineModel('nig-w', (bld) => {
      const sigma2 = bld.latent('sigma2', d.inverseGamma(a, b))
      const mu = bld.latent('mu', d.normal(m0, ops.sqrt(ops.div(sigma2, kappa))))
      bld.observe('y', d.normal(mu, ops.sqrt(ops.div(sigma2, 2))), bld.data('y', { dims: ['i'], role: 'response' }))
    })
    const { marginals } = run(weighted.bind({ y: Y }))
    const W = w.reduce((p, q) => p + q, 0)
    const wy = w.reduce((p, q, i) => p + q * (Y[i] as number), 0)
    const wkn = kappa + W
    const wmn = (kappa * m0 + wy) / wkn
    const wyy = w.reduce((p, q, i) => p + q * (Y[i] as number) ** 2, 0)
    const wbn = b + 0.5 * (kappa * m0 ** 2 + wyy - wkn * wmn ** 2)
    const t = marginals['mu'] as Record<string, number>
    expect(t['m']).toBeCloseTo(wmn, 10)
    expect(t['beta']).toBeCloseTo(wkn, 10)
    expect(t['b']).toBeCloseTo(wbn, 9)
  })

  it('an inverse-gamma variance on its own (a known mean) is a gamma precision, exact', () => {
    const own = defineModel('ig', (bld) => {
      const sigma2 = bld.latent('sigma2', d.inverseGamma(a, b))
      bld.observe('y', d.normal(1, ops.sqrt(sigma2)), bld.data('y', { dims: ['i'], role: 'response' }))
    })
    const { c, g, r } = run(own.bind({ y: Y }))
    const post = factorGraphPosterior(c, g, r.state as MpState)
    const bnk = b + Y.reduce((p, q) => p + (q - 1) ** 2, 0) / 2
    expect(item(post.mean('sigma2'))).toBeCloseTo(bnk / (a + n / 2 - 1), 8)
  })

  it('refuses a variance used as a value, with the idiom named', () => {
    const bad = defineModel('bad', (bld) => {
      const sigma2 = bld.latent('sigma2', d.inverseGamma(a, b))
      bld.observe('y', d.normal(ops.add(sigma2, 1), 1), bld.data('y', { dims: ['i'], role: 'response' }))
    })
    const c = toFactorGraph(bad.bind({ y: Y }))
    expect(c.graph).toBeUndefined()
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/inverse-gamma/)
  })
})
