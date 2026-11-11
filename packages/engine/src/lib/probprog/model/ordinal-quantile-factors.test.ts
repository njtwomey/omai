// Ordinal regression (ordered probit and ordered logistic, the cut points fixed) and quantile regression (the asymmetric Laplace) as
// models, under EP, each against the exact posterior of the intercept and slope on a grid.
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
import { defineModel, factorGraphVerdict, toFactorGraph } from './index.ts'

function ep(bound: Parameters<typeof toFactorGraph>[0]) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-7, damping: 0.3 }), { maxSteps: 400 })
  expect(['converged', 'completed']).toContain(r.status)
  return { c, marginals: marginalsOf(g, r.state as MpState) as Record<string, Record<string, unknown>> }
}

/** The means and sds of both coordinates of a normalised log density on a grid. */
function grid2(
  f: (x: number, y: number) => number,
  [x0, x1]: readonly [number, number],
  [y0, y1]: readonly [number, number],
  n = 400,
): { mean: [number, number]; sd: [number, number] } {
  const hx = (x1 - x0) / n
  const hy = (y1 - y0) / n
  let z = 0
  const s1 = [0, 0]
  const s2 = [0, 0]
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
    }
  }
  const mean: [number, number] = [(s1[0] as number) / z, (s1[1] as number) / z]
  return {
    mean,
    sd: [Math.sqrt((s2[0] as number) / z - mean[0] ** 2), Math.sqrt((s2[1] as number) / z - mean[1] ** 2)],
  }
}

const post = (marginals: Record<string, Record<string, unknown>>, name: string): { mean: number; sd: number } => {
  const m = marginals[name] as Record<string, number>
  return { mean: m['mean'] as number, sd: Math.sqrt(m['variance'] as number) }
}
const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)
const normalPrior = (x: number, s: number): number => -0.5 * (x / s) ** 2
const factorsOf = (c: ReturnType<typeof ep>['c'], ...operators: string[]): number =>
  c.graph?.factors.filter((f) => operators.includes(f.operator)).length ?? 0

const x = [-1.5, -1.1, -0.8, -0.3, 0, 0.4, 0.7, 1.1, 1.4, 1.8]
const cuts = [-0.6, 0.5, 1.4]

describe('an ordinal regression with fixed cut points', () => {
  // Four ordered categories along the covariate.
  const y = [0, 0, 1, 0, 1, 1, 2, 3, 2, 3]
  // The cumulative links in plain arithmetic (the grid calls them millions of times): Phi by Abramowitz and Stegun 26.2.17.
  const phi = (z: number): number => {
    const t = 1 / (1 + 0.2316419 * Math.abs(z))
    const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    const tail = Math.exp(-0.5 * z * z) * 0.3989422804014327 * poly
    return z >= 0 ? 1 - tail : tail
  }
  const linkLog = { orderedProbit: phi, orderedLogistic: (z: number) => 1 / (1 + Math.exp(-z)) }
  for (const family of ['orderedProbit', 'orderedLogistic'] as const) {
    it(`${family}: compiles to ordinal likelihood sites and agrees with the exact posterior of the intercept and slope`, () => {
      const dist = family === 'orderedProbit' ? d.orderedProbit : d.orderedLogistic
      const model = defineModel(family, (m) => {
        const row = m.range('row')
        const a = m.latent('a', d.normal(0, 2))
        const b = m.latent('b', d.normal(0, 2))
        const predictor = a.add(b.mul(m.data('x', { dims: [row], role: 'covariate' })))
        m.observe('y', dist(predictor, cuts), m.data('y', { dims: [row], support: 'count' }))
      })
      const { c, marginals } = ep(model.bind({ x, y }))
      expect(factorsOf(c, 'likelihood-row', 'likelihood-site')).toBe(x.length)
      const lc = linkLog[family]
      const logCategory = (k: number, u: number): number => {
        const hi = k === cuts.length ? 1 : lc((cuts[k] as number) - u)
        const lo = k === 0 ? 0 : lc((cuts[k - 1] as number) - u)
        return Math.log(hi - lo)
      }
      const ref = grid2(
        (a, b) =>
          normalPrior(a, 2) + normalPrior(b, 2) + sum(x.map((xi, i) => logCategory(y[i] as number, a + b * xi))),
        [-5, 5],
        [-2, 7],
      )
      for (const [name, k] of [
        ['a', 0],
        ['b', 1],
      ] as const) {
        const got = post(marginals, name)
        expect(
          Math.abs(got.mean - (ref.mean[k] as number)) / (ref.sd[k] as number),
          `${family} ${name} mean`,
        ).toBeLessThan(0.15)
        expect(got.sd / (ref.sd[k] as number), `${family} ${name} sd`).toBeGreaterThan(0.8)
        expect(got.sd / (ref.sd[k] as number), `${family} ${name} sd`).toBeLessThan(1.2)
      }
    })
  }
  it('refuses random cut points with the reason', () => {
    const model = defineModel('random-cuts', (m) => {
      const row = m.range('row')
      const a = m.latent('a', d.normal(0, 2))
      const c1 = m.latent('c1', d.normal(0, 2))
      const cuts2 = c1.add(m.const([0, 1]))
      m.observe('y', d.orderedProbit(a, cuts2), m.data('y', { dims: [row], support: 'count' }))
    })
    const c = toFactorGraph(model.bind({ y: [0, 1, 2] }))
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/cut points .* random/)
  })
})

describe('quantile regression: the asymmetric Laplace with a linear predictor', () => {
  const y = [-1.2, -0.9, -0.1, -0.7, 0.2, 0.1, 0.9, 1.4, 0.8, 2.1]
  const tau = 0.8
  const scale = 0.5
  it('agrees with the exact posterior of the intercept and slope', () => {
    const model = defineModel('quantile-regression', (m) => {
      const row = m.range('row')
      const a = m.latent('a', d.normal(0, 2))
      const b = m.latent('b', d.normal(0, 2))
      const predictor = a.add(b.mul(m.data('x', { dims: [row], role: 'covariate' })))
      m.observe('y', d.asymmetricLaplace(predictor, scale, tau), m.data('y', { dims: [row], role: 'response' }))
    })
    const { c, marginals } = ep(model.bind({ x, y }))
    expect(factorsOf(c, 'likelihood-row', 'likelihood-site')).toBe(x.length)
    const logAld = (v: number, u: number): number => {
      const r = (v - u) / scale
      return Math.log(tau * (1 - tau)) - Math.log(scale) - r * (tau - (r < 0 ? 1 : 0))
    }
    const ref = grid2(
      (a, b) => normalPrior(a, 2) + normalPrior(b, 2) + sum(x.map((xi, i) => logAld(y[i] as number, a + b * xi))),
      [-1.5, 3],
      [-1, 2.5],
    )
    for (const [name, k] of [
      ['a', 0],
      ['b', 1],
    ] as const) {
      const got = post(marginals, name)
      expect(Math.abs(got.mean - (ref.mean[k] as number)) / (ref.sd[k] as number), `${name} mean`).toBeLessThan(0.15)
      expect(got.sd / (ref.sd[k] as number), `${name} sd`).toBeGreaterThan(0.8)
      expect(got.sd / (ref.sd[k] as number), `${name} sd`).toBeLessThan(1.2)
    }
  })
})
