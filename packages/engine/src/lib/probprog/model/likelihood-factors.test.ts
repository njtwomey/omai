// The numeric likelihoods of the registry (Poisson and negative binomial with a log link, binomial with a logistic or probit
// link, Weibull, log-normal) compiled to factor graphs and run under EP, against the exact posterior of a two-coefficient
// regression on a grid (`gridPosterior` of the compiled program: the library's own log densities, not the rules').
import { describe, expect, it } from 'vitest'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  type MpState,
} from '../../message-passing/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { gridPosterior } from '../validation/index.ts'
import * as d from './dists/index.ts'
import { defineModel, factorGraphPosterior, factorGraphVerdict, ops, toFactorGraph, toProgram } from './index.ts'

const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

const X = [
  [1, -1],
  [1, -0.6],
  [1, -0.2],
  [1, 0.2],
  [1, 0.6],
  [1, 1],
  [1, 0.4],
  [1, -0.8],
]
const COUNTS = [0, 1, 1, 2, 4, 6, 3, 0]
const SUCCESSES = [0, 1, 2, 3, 5, 7, 4, 1]
const TIMES = [0.4, 0.7, 1.1, 1.6, 2.9, 4.4, 2.2, 0.5]

type Builder = Parameters<Parameters<typeof defineModel>[1]>[0]

function regression(
  name: string,
  likelihood: (mean: never, y: never, m: Builder, exposure: never) => void,
  support: 'count' | 'positive' = 'count',
): ReturnType<typeof defineModel> {
  return defineModel(name, (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const Xd = m.data('X', { dims: [row, feature] })
    const E = m.data('E', { dims: [row] })
    const y = m.data('y', { dims: [row], support })
    const w = m.latent('w', d.normal(0, 1), { dims: [feature] })
    likelihood(Xd.matmul(w) as never, y as never, m, E as never)
  })
}

function compare(
  def: ReturnType<typeof defineModel>,
  y: readonly number[],
  tol: { mean: number; sd: [number, number] },
): void {
  const data = { X, y, E: [1, 2, 1, 2, 1, 2, 1, 2] }
  const bound = def.bind(data)
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = graphOf(c.graph)
  expect(factorGraphVerdict(c, g, 'ep')).toMatchObject({ ok: true })
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-9 }), { maxSteps: 400 })
  expect(r.status).toBe('converged')
  const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
  const mean = toFlat(post.mean('w'))
  const sd = post
    .summary()
    .filter((s) => s.site === 'w')
    .map((s) => s.sd)
  const ref = gridPosterior(toProgram(bound), undefined, { points: 61, maxRefinements: 1 })
  for (const k of [0, 1]) {
    const m = ref.marginal(`w[${k}]`)
    expect(
      Math.abs((mean[k] as number) - m.mean) / m.sd,
      `mean of w[${k}] (reference ${m.mean} +- ${m.sd})`,
    ).toBeLessThan(tol.mean)
    const ratio = (sd[k] as number) / m.sd
    expect(ratio, `sd of w[${k}] (reference ${m.sd}, EP ${sd[k]})`).toBeGreaterThan(tol.sd[0])
    expect(ratio).toBeLessThan(tol.sd[1])
  }
}

describe('numeric likelihoods of the registry under expectation propagation', () => {
  it('Poisson with a log link', () => {
    compare(
      regression('poisson-log', (mean, y, m) => m.observe('y', d.poisson(ops.exp(mean)), y)),
      COUNTS,
      {
        mean: 0.15,
        sd: [0.85, 1.2],
      },
    )
  })

  it('Poisson with a log link and an exposure', () => {
    compare(
      regression('poisson-exposure', (mean, y, m, E) => m.observe('y', d.poisson(ops.mul(E, ops.exp(mean))), y)),
      COUNTS,
      { mean: 0.15, sd: [0.85, 1.2] },
    )
  })

  it('binomial with a logistic link and with a probit link', () => {
    compare(
      regression('binomial-logit', (mean, y, m) => m.observe('y', d.binomial(8, ops.sigmoid(mean)), y)),
      SUCCESSES,
      { mean: 0.15, sd: [0.85, 1.2] },
    )
    compare(
      regression('binomial-probit', (mean, y, m) => m.observe('y', d.binomial(8, ops.probit(mean)), y)),
      SUCCESSES,
      { mean: 0.15, sd: [0.85, 1.2] },
    )
  })

  it('negative binomial with a log link and a known dispersion', () => {
    // NB2: r = 2 successes, probability r / (r + mean) (mean = exp(mean), dispersion 1 / 2).
    compare(
      regression('negative-binomial', (mean, y, m) =>
        m.observe('y', d.negativeBinomial(2, ops.div(2, ops.add(2, ops.exp(mean)))), y),
      ),
      COUNTS,
      { mean: 0.15, sd: [0.85, 1.2] },
    )
  })

  it('Weibull survival times with a log link and a known shape', () => {
    compare(
      regression('weibull', (mean, y, m) => m.observe('y', d.weibull(1.5, ops.exp(mean)), y), 'positive'),
      TIMES,
      { mean: 0.15, sd: [0.85, 1.2] },
    )
  })

  it('log-normal survival times: the exact Gaussian rule on the log of the time', () => {
    compare(
      regression('log-normal', (mean, y, m) => m.observe('y', d.logNormal(mean, 0.6), y), 'positive'),
      TIMES,
      { mean: 0.02, sd: [0.97, 1.03] },
    )
  })

  it('a random dispersion or shape is refused with the reason, not mapped wrongly', () => {
    const def = defineModel('nb-random', (m) => {
      const row = m.range('row')
      const feature = m.range('feature')
      const Xd = m.data('X', { dims: [row, feature] })
      const w = m.latent('w', d.normal(0, 1), { dims: [feature] })
      const alpha = m.latent('alpha', d.gamma(2, 2))
      const mean = ops.exp(Xd.matmul(w))
      m.observe(
        'y',
        d.negativeBinomial(ops.div(1, alpha), ops.div(1, ops.add(1, alpha.mul(mean)))),
        m.data('y', { dims: [row], support: 'count' }),
      )
    })
    const c = toFactorGraph(def.bind({ X, y: COUNTS }))
    expect(c.issues.map((i) => i.reason).join(' ')).toMatch(/dispersion/)
  })

  it('negative binomial with a random half-normal dispersion, scalar coefficients packed: the exact posterior of (a, b, alpha) on a grid', () => {
    const def = defineModel('nb-dispersion', (m) => {
      const row = m.range('row')
      const x = m.data('x', { dims: [row] })
      const a = m.latent('a', d.normal(0, 2))
      const b = m.latent('b', d.normal(0, 2))
      const alpha = m.latent('alpha', d.halfNormal(1))
      const mean = ops.exp(a.add(b.mul(x)))
      m.observe(
        'y',
        d.negativeBinomial(ops.div(1, alpha), ops.div(1, ops.add(1, alpha.mul(mean)))),
        m.data('y', { dims: [row], support: 'count' }),
      )
    })
    const data = { x: [-1, -0.6, -0.2, 0.2, 0.6, 1, 0.4, -0.8, 0.9, 0.1], y: [0, 1, 1, 2, 4, 6, 3, 0, 9, 2] }
    const bound = def.bind(data)
    const c = toFactorGraph(bound, { packScalars: true })
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'negative-binomial-dispersion')).toHaveLength(10)
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-7, damping: 0.5 }), { maxSteps: 400 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
    const ref = gridPosterior(toProgram(bound), undefined, { points: 27, maxRefinements: 0 })
    for (const name of ['a', 'b', 'alpha']) {
      const m = ref.marginal(name)
      const got = {
        mean: toFlat(post.mean(name))[0] as number,
        sd: post.summary().find((s) => s.site === name)?.sd as number,
      }
      expect(Math.abs(got.mean - m.mean) / m.sd, `mean of ${name}`).toBeLessThan(0.3)
      expect(got.sd / m.sd, `sd of ${name}`).toBeGreaterThan(0.75)
      // The dispersion is a positive value carried on its log with the exact half-normal prior (a numeric site): its posterior piles up
      // against zero, which one Gaussian on the log holds as a log-normal with a long right tail (spread 1.52 of the grid's; the
      // prior projected once to a Gaussian gave 0.87, by being narrower, while the evidence was off by 0.4 nats). a and b: 0.96.
      expect(got.sd / m.sd, `sd of ${name}`).toBeLessThan(name === 'alpha' ? 1.7 : 1.3)
    }
  }, 60_000)

  it('packScalars: scalar coefficients that share a predictor keep their correlation (the marginal sds of a loop of scalar messages come out too small)', () => {
    const def = defineModel('poisson-scalars', (m) => {
      const row = m.range('row')
      const x = m.data('x', { dims: [row] })
      const a = m.latent('a', d.normal(0, 2))
      const b = m.latent('b', d.normal(0, 2))
      m.observe('y', d.poisson(ops.exp(a.add(b.mul(x)))), m.data('y', { dims: [row], support: 'count' }))
    })
    const bound = def.bind({ x: [0, 0.5, 1, 1.5, 2, 2.5, 3, 0.2], y: [1, 1, 3, 4, 8, 11, 17, 1] })
    const ref = gridPosterior(toProgram(bound), undefined, { points: 81, maxRefinements: 1 })
    const sds = (pack: boolean): Record<string, number> => {
      const c = toFactorGraph(bound, { packScalars: pack })
      const g = graphOf(c.graph)
      const r = runAlgorithm(epAlgorithm(g, { tol: 1e-9, damping: 0.3 }), { maxSteps: 600 })
      const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'ep' })
      return Object.fromEntries(['a', 'b'].map((n) => [n, post.summary().find((s) => s.site === n)?.sd as number]))
    }
    const loose = sds(false)
    const packed = sds(true)
    for (const n of ['a', 'b']) {
      const sd = ref.marginal(n).sd
      expect((packed[n] as number) / sd, `packed sd of ${n}`).toBeGreaterThan(0.9)
      expect((packed[n] as number) / sd, `packed sd of ${n}`).toBeLessThan(1.1)
      expect((loose[n] as number) / sd, `scalar sd of ${n}`).toBeLessThan(0.8)
    }
  })
})
