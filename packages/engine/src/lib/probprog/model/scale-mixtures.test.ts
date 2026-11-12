// Scale mixtures of Gaussians under EP: a Student-t likelihood with an unknown scale (a gamma mixing precision per row times
// the noise precision) and a horseshoe coefficient (half-Cauchy local and global scales as gamma hierarchies), each against
// the exact posterior on a grid.
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../message-passing/index.ts'
import * as d from './dists/index.ts'
import { defineModel, type toFactorGraph } from './index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0], damping = 0.3) {
  const { c, g, r } = runEp(bound, { tol: 1e-9, damping: damping, maxSteps: 800, verdict: true })
  return { c, r, marginals: marginalsOf(g, r.state as MpState) }
}

const lgam = (z: number): number => {
  // Lanczos, independent of the library, for the reference density.
  const c = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  const x = z - 1
  let a = 0.99999999999980993
  for (let i = 0; i < 8; i++) a += (c[i] as number) / (x + i + 1)
  const t = x + 7.5
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

describe('a Student-t likelihood with an unknown scale', () => {
  const nu = 4
  const ys = [0.4, 0.9, 0.2, 0.6, 5.1, 0.7, 0.5]
  const model = defineModel('t-scale', (b) => {
    const row = b.range('row')
    const mu = b.latent('mu', d.normal(0, 3))
    const tau = b.latent('tau', d.gamma(2, 1))
    const u = b.latent('u', d.gamma(nu / 2, nu / 2), { dims: [row] })
    b.observe('y', d.normal(mu, u.mul(tau).pow(-0.5)), b.data('y', { dims: [row] }))
  })

  it('the mixing precisions times the noise precision are gamma-product variables, and EP is close to the exact posterior of the location and the precision', () => {
    const { c, r, marginals } = ep(model.bind({ y: ys }))
    expect(c.graph?.factors.filter((f) => f.operator === 'gamma-product').length).toBe(ys.length)
    expect(['converged', 'completed']).toContain(r.status)
    // The exact posterior of (mu, tau) with the Student-t density, the mixing precisions integrated out.
    const logt = (r2: number, tau: number): number =>
      lgam((nu + 1) / 2) -
      lgam(nu / 2) -
      0.5 * Math.log(nu * Math.PI) +
      0.5 * Math.log(tau) -
      ((nu + 1) / 2) * Math.log(1 + (tau * r2) / nu)
    const nm = 500
    const nt = 500
    let z = 0
    let m1 = 0
    let m2 = 0
    let t1 = 0
    let t2 = 0
    for (let i = 0; i < nm; i++) {
      const mu = -3 + (7 * i) / (nm - 1)
      for (let j = 0; j < nt; j++) {
        const lt = -4 + (7 * j) / (nt - 1)
        const tau = Math.exp(lt)
        let lp = -0.5 * (mu / 3) ** 2 + Math.log(tau) - tau + lt
        for (const y of ys) lp += logt((y - mu) ** 2, tau)
        const w = Math.exp(lp)
        z += w
        m1 += w * mu
        m2 += w * mu * mu
        t1 += w * tau
        t2 += w * tau * tau
      }
    }
    const mean = m1 / z
    const sd = Math.sqrt(m2 / z - mean * mean)
    const tauMean = t1 / z
    const tauSd = Math.sqrt(t2 / z - tauMean * tauMean)
    const got = marginals['mu'] as { mean: number; variance: number }
    const tau = marginals['tau'] as { shape: number; rate: number }
    console.log(
      't-scale',
      got.mean,
      Math.sqrt(got.variance),
      mean,
      sd,
      tau.shape / tau.rate,
      Math.sqrt(tau.shape) / tau.rate,
      tauMean,
      tauSd,
    )
    expect(Math.abs(got.mean - mean)).toBeLessThan(0.15 * sd)
    expect(Math.sqrt(got.variance) / sd).toBeGreaterThan(0.8)
    expect(Math.sqrt(got.variance) / sd).toBeLessThan(1.25)
    expect(Math.abs(tau.shape / tau.rate - tauMean)).toBeLessThan(0.15 * tauSd)
  })
})

describe('a horseshoe coefficient (half-Cauchy local and global scales as gamma hierarchies)', () => {
  // beta | a, g ~ N(0, 1 / (a g)), a | b ~ Gamma(1/2, b), b ~ Gamma(1/2, 1) (a = 1 / lambda^2, lambda half-Cauchy), and the same for g = 1 / tau^2.
  const horseshoe = (n: number) =>
    defineModel('horseshoe', (b) => {
      const row = b.range('row')
      const cg = b.latent('c', d.gamma(0.5, 1))
      const g = b.latent('g', d.gamma(0.5, cg))
      const cl = b.latent('b', d.gamma(0.5, 1))
      const a = b.latent('a', d.gamma(0.5, cl))
      const beta = b.latent('beta', d.normal(0, a.mul(g).pow(-0.5)))
      const x = b.data('x', { dims: [row] })
      void n
      b.observe('y', d.normal(beta.mul(x), 0.5), b.data('y', { dims: [row] }))
    })
  const x = [1, -0.5, 0.8, 1.3, -1.1, 0.4]
  const y = [1.9, -0.8, 1.7, 2.4, -2.3, 0.6]

  it('compiles to gamma-rate and gamma-product factors and EP is close to the exact posterior of the coefficient', () => {
    const { c, r, marginals } = ep(horseshoe(x.length).bind({ x, y }))
    expect(c.graph?.factors.some((f) => f.operator === 'gamma-product')).toBe(true)
    expect(c.graph?.factors.filter((f) => f.operator === 'gamma-rate').length).toBe(2)
    // The exact posterior of (beta, tau): the prior of beta given tau is the half-Cauchy mixture of normals, a one-dimensional integral over lambda.
    const nb = 360
    const nt = 260
    const nl = 400
    const lam = Array.from({ length: nl }, (_, k) => Math.exp(-8 + (16 * k) / (nl - 1)))
    const dl = 16 / (nl - 1)
    let z = 0
    let m1 = 0
    let m2 = 0
    for (let j = 0; j < nt; j++) {
      const lt = -7 + (11 * j) / (nt - 1)
      const tau = Math.exp(lt)
      for (let i = 0; i < nb; i++) {
        const beta = -3.5 + (7 * i) / (nb - 1)
        let prior = 0
        for (const l of lam) {
          const s2 = (l * tau) ** 2
          prior +=
            (Math.exp((-0.5 * beta * beta) / s2) / Math.sqrt(2 * Math.PI * s2)) * (2 / (Math.PI * (1 + l * l))) * l * dl
        }
        let ll = 0
        for (const [k, xv] of x.entries()) ll += -0.5 * (((y[k] as number) - beta * xv) / 0.5) ** 2
        // The density of tau is half-Cauchy; in log tau it carries the Jacobian tau.
        const w = prior * Math.exp(ll) * (2 / (Math.PI * (1 + tau * tau))) * tau
        z += w
        m1 += w * beta
        m2 += w * beta * beta
      }
    }
    const mean = m1 / z
    const sd = Math.sqrt(m2 / z - mean * mean)
    const got = marginals['beta'] as { mean: number; variance: number }
    console.log('horseshoe', r.status, got.mean, Math.sqrt(got.variance), mean, sd)
    expect(Math.abs(got.mean - mean)).toBeLessThan(0.2 * sd)
    expect(Math.sqrt(got.variance) / sd).toBeGreaterThan(0.7)
    expect(Math.sqrt(got.variance) / sd).toBeLessThan(1.4)
  })
})

describe('a sparse regression with a horseshoe prior on six coefficients', () => {
  const P = 6
  const n = 30
  const unit = (i: number, j: number): number => Math.abs((Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1)
  const X = Array.from({ length: n }, (_, i) => Array.from({ length: P }, (_, j) => 3.4 * (unit(i, j) - 0.5)))
  const truth = [2, 0, 0, -1.5, 0, 0.3]
  const y = X.map((r, i) => r.reduce((a, v, j) => a + v * (truth[j] as number), 0) + 1.7 * (unit(i, 99) - 0.5))
  const model = defineModel('horseshoe-regression', (b) => {
    const row = b.range('row')
    const f = b.range('f', { size: P })
    const cg = b.latent('c', d.gamma(0.5, 1))
    const g = b.latent('g', d.gamma(0.5, cg))
    const cl = b.latent('b', d.gamma(0.5, 1), { dims: [f] })
    const a = b.latent('a', d.gamma(0.5, cl), { dims: [f] })
    const beta = b.latent('beta', d.normal(0, a.mul(g).pow(-0.5)), { dims: [f] })
    b.observe('y', d.normal(b.data('X', { dims: [row, f] }).matmul(beta), 0.5), b.data('y', { dims: [row] }))
  })

  it('EP agrees with the means of a long NUTS run (three chains of 600 draws) to about half a posterior sd', () => {
    const { marginals, r } = ep(model.bind({ X, y }), 0.4)
    expect(r.status).toBe('converged')
    // Recorded from the NUTS run (posterior means and sds of beta); the shrunk coefficients are heavy-tailed, so EP's sds there are narrower.
    const nuts = [
      [1.997, 0.095],
      [0.032, 0.085],
      [-0.047, 0.086],
      [-1.398, 0.088],
      [-0.139, 0.098],
      [0.164, 0.095],
    ] as const
    for (const [j, [mean, sd]] of nuts.entries()) {
      const got = marginals[`beta[${j}]`] as { mean: number; variance: number }
      expect(Math.abs(got.mean - mean)).toBeLessThan(0.6 * sd)
      expect(Math.sqrt(got.variance)).toBeGreaterThan(0.35 * sd)
      expect(Math.sqrt(got.variance)).toBeLessThan(1.3 * sd)
    }
  })
})
