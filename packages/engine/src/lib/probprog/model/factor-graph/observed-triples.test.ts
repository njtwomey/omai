// Likelihoods over three random scalars as models, under EP, each against the exact posterior of the unknowns on a grid: a Student-t
// whose location, scale and degrees of freedom are all random (a `likelihood-triple` factor), and an ordinal outcome whose cut points
// are random (an ordered latent vector, or a learned draw margin) beside the slope.
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../../message-passing/index.ts'
import { lgammaOf, logPhi } from '../../../message-passing/numerics/_scalar.ts'
import * as d from '../dists/index.ts'
import { defineModel, ops, type toFactorGraph } from '../index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-6, damping: 0.3, maxSteps: 300, verdict: true, converged: true })
  return { c, marginals: marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>> }
}

/** The mean and sd of each coordinate of a normalised log density on a midpoint grid of `n` cells per coordinate. */
function grid3(
  f: (x: number, y: number, z: number) => number,
  box: readonly [readonly [number, number], readonly [number, number], readonly [number, number]],
  n: number,
): { mean: number[]; sd: number[]; value: number[]; valueSd: number[] } {
  const h = box.map(([lo, hi]) => (hi - lo) / n)
  let total = 0
  const m1 = [0, 0, 0]
  const m2 = [0, 0, 0]
  const e1 = [0, 0, 0]
  const e2 = [0, 0, 0]
  for (let i = 0; i < n; i++) {
    const x = box[0][0] + (i + 0.5) * (h[0] as number)
    for (let j = 0; j < n; j++) {
      const y = box[1][0] + (j + 0.5) * (h[1] as number)
      for (let k = 0; k < n; k++) {
        const z = box[2][0] + (k + 0.5) * (h[2] as number)
        const w = Math.exp(f(x, y, z))
        total += w
        for (const [q, v] of [x, y, z].entries()) {
          m1[q] = (m1[q] as number) + w * v
          m2[q] = (m2[q] as number) + w * v * v
          e1[q] = (e1[q] as number) + w * Math.exp(v)
          e2[q] = (e2[q] as number) + w * Math.exp(2 * v)
        }
      }
    }
  }
  const mean = m1.map((v) => v / total)
  const value = e1.map((v) => v / total)
  return {
    mean,
    sd: m2.map((v, q) => Math.sqrt(Math.max(v / total - (mean[q] as number) ** 2, 0))),
    value,
    valueSd: e2.map((v, q) => Math.sqrt(Math.max(v / total - (value[q] as number) ** 2, 0))),
  }
}

const post = (marginals: Record<string, Record<string, unknown>>, name: string): { mean: number; sd: number } => {
  const m = marginals[name] as Record<string, number>
  return { mean: m['mean'] as number, sd: Math.sqrt(m['variance'] as number) }
}
/** The mean within `z` reference sds and the sd within the ratio band. */
function agree(
  got: { mean: number; sd: number },
  mean: number,
  sd: number,
  z: number,
  band: number,
  what: string,
): void {
  expect(Math.abs(got.mean - mean) / sd, `${what} mean ${got.mean} against ${mean}`).toBeLessThan(z)
  expect(got.sd / sd, `${what} sd ${got.sd} against ${sd}`).toBeGreaterThan(1 - band)
  expect(got.sd / sd, `${what} sd ${got.sd} against ${sd}`).toBeLessThan(1 + band)
}
const factorsOf = (c: ReturnType<typeof ep>['c'], operator: string): number =>
  c.graph?.factors.filter((f) => f.operator === operator).length ?? 0
const normalLog = (x: number, m: number, s: number): number => -0.5 * ((x - m) / s) ** 2
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

describe('a Student-t likelihood whose location, scale and degrees of freedom are all random', () => {
  const y = [0.3, -0.4, 0.1, 0.8, -0.2, 3.9, 0.5, -2.8, 0.2, -0.6, 0.9, 0.0]
  const model = defineModel('student-t-full', (b) => {
    const mu = b.latent('mu', d.normal(0, 2))
    const scale = b.latent('scale', d.logNormal(-0.2, 0.7))
    const df = b.latent('df', d.logNormal(1.5, 0.8))
    b.observe('y', d.studentT(df, mu, scale), b.data('y', { dims: ['i'], role: 'response' }))
  })
  it('compiles to likelihood-triple factors; EP agrees with the exact posterior of the three unknowns', () => {
    const { c, marginals } = ep(model.bind({ y }))
    expect(c.issues).toEqual([])
    expect(factorsOf(c, 'likelihood-triple')).toBe(y.length)
    const logT = (v: number, mu: number, s: number, df: number): number =>
      lgammaOf((df + 1) / 2) -
      lgammaOf(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      s -
      ((df + 1) / 2) * Math.log1p(((v - mu) * Math.exp(-s)) ** 2 / df)
    const ref = grid3(
      (mu, s, t) =>
        normalLog(mu, 0, 2) +
        normalLog(s, -0.2, 0.7) +
        normalLog(t, 1.5, 0.8) +
        sum(y.map((v) => logT(v, mu, s, Math.exp(t)))),
      [
        [-1, 1.5],
        [-2.5, 1.5],
        [-0.5, 4.5],
      ],
      90,
    )
    agree(post(marginals, 'mu'), ref.mean[0] as number, ref.sd[0] as number, 0.2, 0.25, 'mu')
    // the scale and the degrees of freedom are positive: compared as values (the moments of exp of the coordinate)
    agree(post(marginals, 'scale'), ref.value[1] as number, ref.valueSd[1] as number, 0.3, 0.35, 'scale')
    // (the degrees of freedom have a long right tail that one Gaussian on the log carries at about 0.6 of its spread)
    agree(post(marginals, 'df'), ref.value[2] as number, ref.valueSd[2] as number, 0.3, 0.5, 'df')
  })
})

describe('an ordinal outcome with random cut points', () => {
  const x = [-1.6, -1.2, -0.9, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0, 1.4, 1.7, 2.0]
  const y = [0, 0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 2]
  it('an ordered latent vector of cut points: likelihood-triple factors, EP agrees with the exact posterior of the slope and the cuts', () => {
    const model = defineModel('ordinal-cuts', (m) => {
      const row = m.range('row')
      const cut = m.range('cut', { size: 2 })
      const slope = m.latent('slope', d.normal(0, 2))
      const cuts = m.latent('cuts', d.normal(0, 2), { dims: [cut], constraint: { kind: 'ordered' } })
      const predictor = slope.mul(m.data('x', { dims: [row], role: 'covariate' }))
      m.observe('y', d.orderedProbit(predictor, cuts, { over: cut }), m.data('y', { dims: [row], support: 'count' }))
    })
    const { c, marginals } = ep(model.bind({ x, y }))
    expect(c.issues).toEqual([])
    expect(factorsOf(c, 'likelihood-triple')).toBe(x.length)
    const ref = grid3(
      (slope, c1, c2) => {
        if (!(c1 < c2)) return Number.NEGATIVE_INFINITY
        let lp = normalLog(slope, 0, 2) + normalLog(c1, 0, 2) + normalLog(c2, 0, 2)
        for (const [i, xi] of x.entries()) {
          const u = slope * xi
          const k = y[i] as number
          const lo = k === 0 ? Number.NEGATIVE_INFINITY : k === 1 ? c1 : c2
          const hi = k === 2 ? Number.POSITIVE_INFINITY : k === 0 ? c1 : c2
          const upper = hi === Number.POSITIVE_INFINITY ? 1 : Math.exp(logPhi(hi - u))
          const lower = lo === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logPhi(lo - u))
          lp += Math.log(upper - lower)
        }
        return lp
      },
      [
        [-0.5, 5.5],
        [-3, 1.5],
        [-0.5, 4.5],
      ],
      70,
    )
    // the slope and the cuts share every row (a loop): the means are right and the spreads are 0.75 to 0.95 of the exact ones
    agree(post(marginals, 'slope'), ref.mean[0] as number, ref.sd[0] as number, 0.3, 0.35, 'slope')
    agree(post(marginals, 'cuts[0]'), ref.mean[1] as number, ref.sd[1] as number, 0.3, 0.35, 'cut 0')
    agree(post(marginals, 'cuts[1]'), ref.mean[2] as number, ref.sd[2] as number, 0.3, 0.35, 'cut 1')
  })
  it('a learned draw margin: cut points that are one latent times a known sign vector', () => {
    const model = defineModel('ordinal-margin', (m) => {
      const row = m.range('row')
      const cut = m.range('cut', { size: 2 })
      const slope = m.latent('slope', d.normal(0, 2))
      const margin = m.latent('margin', d.normal(1, 0.7))
      const sign = m.data('sign', { dims: [cut] })
      const predictor = slope.mul(m.data('x', { dims: [row], role: 'covariate' }))
      const cuts = ops.div(margin.mul(sign), 1)
      m.observe('y', d.orderedProbit(predictor, cuts, { over: cut }), m.data('y', { dims: [row], support: 'count' }))
    })
    const { c, marginals } = ep(model.bind({ x, y, sign: [-0.5, 0.5] }))
    expect(c.issues).toEqual([])
    expect(factorsOf(c, 'likelihood-triple')).toBe(x.length)
    expect(post(marginals, 'margin').sd).toBeGreaterThan(0)
  })
})
