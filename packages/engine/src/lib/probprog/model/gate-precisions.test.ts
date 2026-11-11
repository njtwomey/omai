// A gate whose components have precisions of their own: a Gaussian observation whose mean and precision are chosen by the same
// discrete variable (the clutter problem with an unknown inlier and clutter precision, a mixture of Gaussians with unknown
// means and precisions) compiles to one gate of `gaussian-gamma-precision` branches over the means and the gamma variables, and
// EP is close to the exact posterior (the switches summed out, a brute-force grid over the continuous unknowns).
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toFactorGraph } from './index.ts'

function run(bound: Parameters<typeof toFactorGraph>[0], damping = 0.5) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-8, damping }), { maxSteps: 400 })
  return { c, r, marginals: marginalsOf(g, r.state as MpState) }
}

const lognormal = (x: number, mean: number, tau: number): number =>
  0.5 * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (x - mean) ** 2
const logGamma = (t: number, a: number, b: number): number =>
  a * Math.log(b) - lgamma(a) + (a - 1) * Math.log(t) - b * t
const logBeta = (p: number, a: number, b: number): number =>
  lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p)
function lgamma(z: number): number {
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

/** Moments of a coordinate under normalised weights. */
function moments(xs: readonly number[], w: readonly number[]): { mean: number; sd: number } {
  const z = w.reduce((a, b) => a + b, 0)
  const mean = xs.reduce((a, x, i) => a + (x * (w[i] as number)) / z, 0)
  return { mean, sd: Math.sqrt(xs.reduce((a, x, i) => a + ((x - mean) ** 2 * (w[i] as number)) / z, 0)) }
}

describe('the clutter problem with an unknown inlier precision, clutter precision and clutter weight', () => {
  const y = [1.4, 1.6, 1.2, -3.1, 1.7, 1.5, 5.2, 1.1, 1.8, 1.3, 0.9, 1.6]
  const prior = { theta: 5, inlier: [2, 1], clutter: [2, 20], weight: [2, 5] } as const
  const model = defineModel('clutter-learned-test', (m) => {
    const row = m.range('i')
    const theta = m.latent('theta', d.normal(0, prior.theta))
    const inlier = m.latent('inlier', d.gamma(prior.inlier[0], prior.inlier[1]))
    const clutter = m.latent('clutter', d.gamma(prior.clutter[0], prior.clutter[1]))
    const weight = m.latent('weight', d.beta(prior.weight[0], prior.weight[1]))
    const z = m.latent('z', d.bernoulli({ probs: weight }), { dims: [row] })
    const precision = ops.where(z, clutter, inlier)
    m.observe('y', d.normal(ops.where(z, 0, theta), ops.div(1, ops.sqrt(precision))), m.data('y', { dims: [row] }))
  })

  it('compiles to one gate of gaussian-gamma-precision branches per observation, over the centre and both precisions', () => {
    const { c, marginals } = run(model.bind({ y }))
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(y.length)
    const branches = ((gates[0]?.params ?? {}) as { branches?: { operator: string }[] }).branches ?? []
    expect(branches.map((b) => b.operator)).toEqual(['gaussian-gamma-precision', 'gaussian-gamma-precision'])
    expect((marginals['theta'] as { mean: number }).mean).toBeGreaterThan(1)
  })

  it('EP is close to the exact posterior of the centre, the precisions and the weight (a four-dimensional grid, the switches summed out)', () => {
    const { marginals } = run(model.bind({ y }))
    // The grid: centre, log inlier precision, log clutter precision, weight.
    const nt = 36
    const ni = 36
    const nc = 36
    const nw = 36
    const th = Array.from({ length: nt }, (_, i) => 0.6 + (1.8 * i) / (nt - 1))
    const li = Array.from({ length: ni }, (_, i) => -1 + (4.2 * i) / (ni - 1))
    const lc = Array.from({ length: nc }, (_, i) => -4.5 + (4.5 * i) / (nc - 1))
    const wg = Array.from({ length: nw }, (_, i) => 0.02 + (0.7 * i) / (nw - 1))
    const tI = li.map(Math.exp)
    const tC = lc.map(Math.exp)
    const mt = new Array<number>(nt).fill(0)
    const mi = new Array<number>(ni).fill(0)
    const mc = new Array<number>(nc).fill(0)
    const mw = new Array<number>(nw).fill(0)
    const logs: number[] = []
    let top = Number.NEGATIVE_INFINITY
    const idx: [number, number, number, number][] = []
    for (let a = 0; a < nt; a++)
      for (let b = 0; b < ni; b++)
        for (let c = 0; c < nc; c++)
          for (let e = 0; e < nw; e++) {
            let lp =
              lognormal(th[a] as number, 0, 1 / prior.theta ** 2) +
              logGamma(tI[b] as number, prior.inlier[0], prior.inlier[1]) +
              (li[b] as number) +
              logGamma(tC[c] as number, prior.clutter[0], prior.clutter[1]) +
              (lc[c] as number) +
              logBeta(wg[e] as number, prior.weight[0], prior.weight[1])
            for (const v of y) {
              const a1 = Math.log(1 - (wg[e] as number)) + lognormal(v, th[a] as number, tI[b] as number)
              const a2 = Math.log(wg[e] as number) + lognormal(v, 0, tC[c] as number)
              lp += Math.max(a1, a2) + Math.log(1 + Math.exp(-Math.abs(a1 - a2)))
            }
            logs.push(lp)
            idx.push([a, b, c, e])
            if (lp > top) top = lp
          }
    logs.forEach((lp, k) => {
      const w = Math.exp(lp - top)
      const [a, b, c, e] = idx[k] as [number, number, number, number]
      mt[a] = (mt[a] as number) + w
      mi[b] = (mi[b] as number) + w
      mc[c] = (mc[c] as number) + w
      mw[e] = (mw[e] as number) + w
    })
    const exactTheta = moments(th, mt)
    const exactWeight = moments(wg, mw)
    const exactInlier = moments(tI, mi)
    const exactClutter = moments(tC, mc)
    const got = (name: string): { mean: number; sd: number } => {
      const b = marginals[name] as Record<string, number>
      return b['variance'] === undefined
        ? {
            mean: (b['shape'] as number) / (b['rate'] as number),
            sd: Math.sqrt(b['shape'] as number) / (b['rate'] as number),
          }
        : { mean: b['mean'] as number, sd: Math.sqrt(b['variance']) }
    }
    const weight = marginals['weight'] as { mean: number; variance: number }
    const near = (g: { mean: number; sd: number }, e: { mean: number; sd: number }): void => {
      expect(Math.abs(g.mean - e.mean)).toBeLessThan(0.35 * e.sd)
      expect(g.sd / e.sd).toBeGreaterThan(0.6)
      expect(g.sd / e.sd).toBeLessThan(1.5)
    }
    near(got('theta'), exactTheta)
    near(got('inlier'), exactInlier)
    near(got('clutter'), exactClutter)
    near({ mean: weight.mean, sd: Math.sqrt(weight.variance) }, exactWeight)
  })

  it('a known clutter precision and an unknown inlier precision mix a gaussian branch and a gaussian-gamma-precision branch', () => {
    const known = defineModel('clutter-known-precision', (m) => {
      const row = m.range('i')
      const theta = m.latent('theta', d.normal(0, 5))
      const inlier = m.latent('inlier', d.gamma(2, 1))
      const z = m.latent('z', d.bernoulli({ probs: 0.25 }), { dims: [row] })
      const precision = ops.where(z, 0.0625, inlier)
      m.observe('y', d.normal(ops.where(z, 0, theta), ops.div(1, ops.sqrt(precision))), m.data('y', { dims: [row] }))
    })
    const { c, marginals } = run(known.bind({ y }))
    const gate = c.graph?.factors.find((f) => f.operator === 'gate')
    const branches = ((gate?.params ?? {}) as { branches?: { operator: string }[] }).branches ?? []
    expect(branches.map((b) => b.operator)).toEqual(['gaussian-gamma-precision', 'gaussian'])
    expect((marginals['theta'] as { mean: number }).mean).toBeGreaterThan(1)
  })
})

describe('a mixture of two Gaussians with unknown means and precisions, chosen by a categorical switch', () => {
  const x = [-2.1, -1.8, -2.4, -1.5, 2.2, 1.9, 3.1, 2.4, -2.0, 2.6, -1.7, 2.0]
  const model = defineModel('mixture-learned-test', (m) => {
    const row = m.range('i')
    const comp = m.range('component', { size: 2 })
    const mu = m.latent('mu', d.normal(0, 3), { dims: [comp] })
    const tau = m.latent('tau', d.gamma(2, 1), { dims: [comp] })
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    m.observe('x', d.normal(mu.at(z), ops.div(1, ops.sqrt(tau.at(z)))), m.data('x', { dims: [row] }))
  })

  it('compiles to a gate per observation whose branches read the means and the precisions of their own component', () => {
    const { c, marginals } = run(model.bind({ x }), 0.5)
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(x.length)
    const m0 = marginals['mu[0]'] as { mean: number }
    const m1 = marginals['mu[1]'] as { mean: number }
    // The labels can settle either way (the model is symmetric under swapping the components); the two centres are apart.
    expect(Math.abs(m0.mean - m1.mean)).toBeGreaterThan(2)
  })
})
