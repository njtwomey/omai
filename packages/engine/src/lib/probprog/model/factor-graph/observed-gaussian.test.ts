// The Gaussian likelihood on a regression: the regression-row operator and the plated factor, EP against the closed-
// form posterior and evidence.
import { describe, expect, it } from 'vitest'
import { multivariateNormal } from '../../../dists/index.ts'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../../message-passing/index.ts'
import { inv } from '../../../numerics/linalg/index.ts'
import { toArray, toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import { defineModel, factorGraphPosterior, toFactorGraph } from '../index.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

const _N = 40

describe('a conjugate Gaussian regression: EP equals the closed-form posterior and evidence', () => {
  const rows = 12
  const x = Array.from({ length: rows }, (_, i) => [1, Math.sin(i), (i % 4) - 1.5])
  const yv = x.map(
    (r, i) => 0.5 * (r[0] as number) - 1 * (r[1] as number) + 2 * (r[2] as number) + 0.3 * Math.cos(3 * i),
  )
  const sigma = 0.5
  const tau = 3
  const model = defineModel('linear', (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f], role: 'covariate' })
    const y = m.data('y', { dims: [n], role: 'response' })
    const beta = m.latent('beta', d.normal(0, tau), { dims: [f] })
    m.observe('y', d.normal(X.matmul(beta), sigma), y)
  })
  const c = toFactorGraph(model.bind({ X: x, y: yv }))
  it('uses the regression-row operator, a vector variable and no auxiliary (expanded), or one plated factor on sufficient statistics', () => {
    const e = toFactorGraph(model.bind({ X: x, y: yv }), { plates: 'expand' })
    expect(e.issues).toEqual([])
    expect(new Set(e.graph?.factors.map((f) => f.operator))).toEqual(new Set(['prior', 'linear-regression-row']))
    expect(e.auxiliary).toEqual([])
    expect(e.plated).toEqual([])
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => `${f.id} ${f.operator}`)).toEqual([
      'factor:beta prior',
      'factor:y linear-regression-plate',
    ])
    expect(c.graph?.variables.map((v) => v.name)).toEqual(['beta'])
    expect(c.plated).toEqual(['y'])
    expect(c.sites.find((s) => s.site === 'y')).toMatchObject({ layout: 'observed', posterior: false })
    // The expanded graph has the same posterior and evidence as the plated one.
    const ga = graphOf(c.graph)
    const gb = graphOf(e.graph)
    const pa = factorGraphPosterior(c, ga, run(ga).state)
    const pb = factorGraphPosterior(e, gb, run(gb).state)
    for (const [q, v] of toFlat(pa.mean('beta')).entries())
      expect(Math.abs(v - (toFlat(pb.mean('beta'))[q] as number))).toBeLessThan(1e-9)
    expect(Math.abs((pa.logEvidence?.value ?? 0) - (pb.logEvidence?.value ?? 1))).toBeLessThan(1e-9)
  })
  it('matches the posterior N((I / tau^2 + X^T X / sigma^2)^-1 X^T y / sigma^2, ...) and the marginal likelihood', () => {
    const g = graphOf(c.graph)
    const r = run(g)
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state)
    const precision = [0, 1, 2].map((i) =>
      [0, 1, 2].map(
        (j) =>
          x.reduce((a, row) => a + (row[i] as number) * (row[j] as number), 0) / sigma ** 2 +
          (i === j ? 1 / tau ** 2 : 0),
      ),
    )
    const cov = toArray(inv(precision)) as number[][]
    const xty = [0, 1, 2].map(
      (j) => x.reduce((a, row, i) => a + (row[j] as number) * (yv[i] as number), 0) / sigma ** 2,
    )
    const mean = cov.map((row) => row.reduce((a, v, j) => a + v * (xty[j] as number), 0))
    const got = toFlat(post.mean('beta'))
    const sd = toFlat(post.stddev('beta'))
    for (let q = 0; q < 3; q++) {
      expect(Math.abs((got[q] as number) - (mean[q] as number))).toBeLessThan(1e-9)
      expect(Math.abs((sd[q] as number) - Math.sqrt((cov[q] as number[])[q] as number))).toBeLessThan(1e-9)
    }
    // y ~ N(0, sigma^2 I + tau^2 X X^T)
    const k = x.map((a) =>
      x.map((b) => tau ** 2 * a.reduce((s, v, j) => s + v * (b[j] as number), 0) + (a === b ? sigma ** 2 : 0)),
    )
    const exact = Number(toFlat(multivariateNormal(new Array<number>(rows).fill(0), { covariance: k }).logProb(yv))[0])
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - exact)).toBeLessThan(1e-8)
  })
})
