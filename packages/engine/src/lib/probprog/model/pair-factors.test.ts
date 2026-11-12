// A likelihood over a predictor and a second random scalar (a Laplace with a random scale, a Student-t with random degrees of
// freedom, a Weibull with a random shape, a beta outcome with two random shapes) and a beta variable with two random shapes, as
// models, under EP, each against the exact posterior of the two unknowns on a grid (the second coordinate on its log).
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../message-passing/index.ts'
import { lgammaOf } from '../../message-passing/numerics/_scalar.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, type toFactorGraph } from './index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0]) {
  const { c, g, r } = runEp(bound, { tol: 1e-7, damping: 0.3, maxSteps: 400, verdict: true, converged: true })
  return { c, marginals: marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>> }
}

/** The moments of both coordinates, and of the value `exp(x)` of each, of a normalised log density on a grid. */
function grid2(
  f: (x: number, y: number) => number,
  [x0, x1]: readonly [number, number],
  [y0, y1]: readonly [number, number],
  n = 600,
): { mean: [number, number]; sd: [number, number]; value: [number, number]; valueSd: [number, number] } {
  const hx = (x1 - x0) / n
  const hy = (y1 - y0) / n
  let z = 0
  const s1 = [0, 0]
  const s2 = [0, 0]
  const e1 = [0, 0]
  const e2 = [0, 0]
  for (let i = 0; i < n; i++) {
    const x = x0 + (i + 0.5) * hx
    for (let j = 0; j < n; j++) {
      const y = y0 + (j + 0.5) * hy
      const w = Math.exp(f(x, y))
      z += w
      s1[0] = (s1[0] as number) + w * x
      s1[1] = (s1[1] as number) + w * y
      s2[0] = (s2[0] as number) + w * x * x
      s2[1] = (s2[1] as number) + w * y * y
      e1[0] = (e1[0] as number) + w * Math.exp(x)
      e1[1] = (e1[1] as number) + w * Math.exp(y)
      e2[0] = (e2[0] as number) + w * Math.exp(2 * x)
      e2[1] = (e2[1] as number) + w * Math.exp(2 * y)
    }
  }
  const at = (a: number[], k: number): number => (a[k] as number) / z
  const sd = (m1: number, m2: number): number => Math.sqrt(Math.max(m2 - m1 * m1, 0))
  return {
    mean: [at(s1, 0), at(s1, 1)],
    sd: [sd(at(s1, 0), at(s2, 0)), sd(at(s1, 1), at(s2, 1))],
    value: [at(e1, 0), at(e1, 1)],
    valueSd: [sd(at(e1, 0), at(e2, 0)), sd(at(e1, 1), at(e2, 1))],
  }
}

const logNormalPdf = (x: number, m: number, s: number): number => -0.5 * ((x - m) / s) ** 2 - Math.log(s)
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/** EP's mean and sd of a scalar site. */
function post(marginals: Record<string, Record<string, unknown>>, name: string): { mean: number; sd: number } {
  const m = marginals[name] as Record<string, number>
  return { mean: m['mean'] as number, sd: Math.sqrt(m['variance'] as number) }
}

/** The mean within `z` reference sds and the sd within the ratio band. */
function agree(got: { mean: number; sd: number }, mean: number, sd: number, z = 0.2, band = 0.2): void {
  expect(Math.abs(got.mean - mean) / sd).toBeLessThan(z)
  expect(got.sd / sd).toBeGreaterThan(1 - band)
  expect(got.sd / sd).toBeLessThan(1 + band)
}

const factorsOf = (c: ReturnType<typeof ep>['c'], operator: string): number =>
  c.graph?.factors.filter((f) => f.operator === operator).length ?? 0

describe('a Laplace likelihood with a random scale', () => {
  const y = [1.2, 0.7, 1.9, 1.1, 0.4, 1.6, 2.4, 0.9]
  const model = defineModel('laplace-random-scale', (b) => {
    const mu = b.latent('mu', d.normal(0, 3))
    const scale = b.latent('scale', d.logNormal(-0.3, 0.6))
    b.observe('y', d.laplace(mu, scale), b.data('y', { dims: ['i'], role: 'response' }))
  })
  it('compiles to likelihood-pair factors and agrees with the exact posterior of the location and the scale', () => {
    const { c, marginals } = ep(model.bind({ y }))
    expect(factorsOf(c, 'likelihood-pair')).toBe(y.length)
    const ref = grid2(
      (mu, l) =>
        logNormalPdf(mu, 0, 3) +
        logNormalPdf(l, -0.3, 0.6) +
        sum(y.map((v) => -Math.LN2 - l - Math.abs(v - mu) * Math.exp(-l))),
      [-1, 4],
      [-4, 2.5],
    )
    agree(post(marginals, 'mu'), ref.mean[0], ref.sd[0])
    agree(post(marginals, 'scale'), ref.value[1], ref.valueSd[1], 0.2, 0.25)
  })
})

describe('a Student-t likelihood with random degrees of freedom', () => {
  const y = [0.3, -0.4, 0.1, 0.8, -0.2, 4.2, 0.5, -3.1, 0.2, -0.6]
  const model = defineModel('student-t-df', (b) => {
    const mu = b.latent('mu', d.normal(0, 2))
    const df = b.latent('df', d.logNormal(1.5, 0.8))
    b.observe('y', d.studentT(df, mu, 1), b.data('y', { dims: ['i'], role: 'response' }))
  })
  it('a log-normal df is a port of likelihood-pair; EP agrees with the exact posterior of the location and the df', () => {
    const { c, marginals } = ep(model.bind({ y }))
    expect(factorsOf(c, 'likelihood-pair')).toBe(y.length)
    const logT = (v: number, mu: number, df: number): number =>
      lgammaOf((df + 1) / 2) -
      lgammaOf(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      ((df + 1) / 2) * Math.log1p((v - mu) ** 2 / df)
    const ref = grid2(
      (mu, t) => logNormalPdf(mu, 0, 2) + logNormalPdf(t, 1.5, 0.8) + sum(y.map((v) => logT(v, mu, Math.exp(t)))),
      [-1.5, 1.5],
      [-1, 5.5],
    )
    agree(post(marginals, 'mu'), ref.mean[0], ref.sd[0])
    agree(post(marginals, 'df'), ref.value[1], ref.valueSd[1], 0.25, 0.3)
  })
})

describe('a Weibull likelihood with a random shape', () => {
  const y = [2.1, 0.7, 1.4, 3.0, 1.1, 0.9, 1.8, 2.6]
  const model = defineModel('weibull-shape', (b) => {
    const logScale = b.latent('logScale', d.normal(0.3, 1))
    const shape = b.latent('shape', d.logNormal(0.3, 0.7))
    b.observe('y', d.weibull(shape, ops.exp(logScale)), b.data('y', { dims: ['i'], role: 'response' }))
  })
  it('agrees with the exact posterior of the log scale and the shape', () => {
    const { c, marginals } = ep(model.bind({ y }))
    expect(factorsOf(c, 'likelihood-pair')).toBe(y.length)
    const logW = (v: number, u: number, k: number): number =>
      Math.log(k) + (k - 1) * Math.log(v) - k * u - (v / Math.exp(u)) ** k
    const ref = grid2(
      (u, t) => logNormalPdf(u, 0.3, 1) + logNormalPdf(t, 0.3, 0.7) + sum(y.map((v) => logW(v, u, Math.exp(t)))),
      [-1, 2],
      [-1.5, 2.5],
    )
    agree(post(marginals, 'logScale'), ref.mean[0], ref.sd[0])
    agree(post(marginals, 'shape'), ref.value[1], ref.valueSd[1], 0.2, 0.25)
  })
})

describe('beta outcomes with two random shapes', () => {
  const y = [0.12, 0.3, 0.22, 0.5, 0.18, 0.27, 0.41, 0.09]
  const model = defineModel('beta-outcomes', (b) => {
    const a = b.latent('a', d.logNormal(0.5, 0.7))
    const c = b.latent('b', d.logNormal(1, 0.7))
    b.observe('y', d.beta(a, c), b.data('y', { dims: ['i'], role: 'response' }))
  })
  it('agrees with the exact posterior of the two shapes', () => {
    const { c, marginals } = ep(model.bind({ y }))
    expect(factorsOf(c, 'likelihood-pair')).toBe(y.length)
    const ref = grid2(
      (ta, tb) => {
        const [a, b] = [Math.exp(ta), Math.exp(tb)]
        return (
          logNormalPdf(ta, 0.5, 0.7) +
          logNormalPdf(tb, 1, 0.7) +
          sum(
            y.map(
              (v) => (a - 1) * Math.log(v) + (b - 1) * Math.log1p(-v) + lgammaOf(a + b) - lgammaOf(a) - lgammaOf(b),
            ),
          )
        )
      },
      [-1.5, 3.5],
      [-1, 4.5],
    )
    // The two shapes are strongly correlated in the posterior (the data fix their ratio better than their size), and one Gaussian
    // per shape cannot hold the correlation: the means are right and the spreads are 0.65 to 0.8 of the exact ones.
    agree(post(marginals, 'a'), ref.value[0], ref.valueSd[0], 0.2, 0.4)
    agree(post(marginals, 'b'), ref.value[1], ref.valueSd[1], 0.2, 0.4)
  })
})

describe('a beta variable with two random shapes', () => {
  const flips = [1, 0, 1, 1, 1, 0, 1, 1]
  const model = defineModel('beta-two-shapes', (b) => {
    const a = b.latent('a', d.logNormal(0.5, 0.7))
    const c = b.latent('b', d.logNormal(0.5, 0.7))
    const p = b.latent('p', d.beta(a, c))
    b.observe('y', d.bernoulli({ probs: p }), b.data('y', { dims: ['i'], support: 'binary' }))
  })
  it('compiles to one beta-two-shapes factor; p and the two shapes match the exact posterior', () => {
    const { c, marginals } = ep(model.bind({ y: flips }))
    expect(factorsOf(c, 'beta-two-shapes')).toBe(1)
    const k = sum(flips)
    const n = flips.length
    // The conjugate marginal of the data given (a, b) is B(a + k, b + n - k) / B(a, b).
    const lb = (x: number, y2: number): number => lgammaOf(x) + lgammaOf(y2) - lgammaOf(x + y2)
    const logPost = (ta: number, tb: number): number => {
      const [a, b] = [Math.exp(ta), Math.exp(tb)]
      return logNormalPdf(ta, 0.5, 0.7) + logNormalPdf(tb, 0.5, 0.7) + lb(a + k, b + n - k) - lb(a, b)
    }
    const ref = grid2(logPost, [-3, 3.5], [-3, 3.5], 700)
    // E p = E[(a + k) / (a + b + n)] and E p^2 under that posterior.
    const m = 700
    const h = 6.5 / m
    let z = 0
    let e1 = 0
    let e2 = 0
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) {
        const [a, b] = [Math.exp(-3 + (i + 0.5) * h), Math.exp(-3 + (j + 0.5) * h)]
        const w = Math.exp(logPost(-3 + (i + 0.5) * h, -3 + (j + 0.5) * h))
        const [s, t] = [a + k, b + n - k]
        z += w
        e1 += (w * s) / (s + t)
        e2 += (w * (s * (s + 1))) / ((s + t) * (s + t + 1))
      }
    const p = post(marginals, 'p')
    expect(Math.abs(p.mean - e1 / z)).toBeLessThan(0.01)
    expect(p.sd / Math.sqrt(e2 / z - (e1 / z) ** 2)).toBeGreaterThan(0.85)
    expect(p.sd / Math.sqrt(e2 / z - (e1 / z) ** 2)).toBeLessThan(1.15)
    agree(post(marginals, 'a'), ref.value[0], ref.valueSd[0], 0.25, 0.3)
    agree(post(marginals, 'b'), ref.value[1], ref.valueSd[1], 0.25, 0.3)
  })
})
