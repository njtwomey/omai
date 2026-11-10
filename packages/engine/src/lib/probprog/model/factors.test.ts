// A bound spec compiled to a factor graph for message passing: the plates are expanded with the ids of `ids.ts`, the
// engine's EP reproduces the exact answers (closed forms where the model is Gaussian or conjugate, an exact grid for the
// probit regression, the TrueSkill update of `pgm`), the same graph is read back as a posterior by site name, the ids
// agree with the drawing of the spec, and what cannot be mapped is refused with the node and the reason.
import { describe, expect, it } from 'vitest'
import { probitData, probitModel as probitDefinition, probitGrid } from '../../../testing/probit-slice.ts'
import { multivariateNormal } from '../../dists/index.ts'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  type FactorGraphSpec,
  defaultOperators as factorOperators,
  gaussianMvPrior,
  type MpState,
  marginalsOf,
  vmpAlgorithm,
} from '../../message-passing/index.ts'
import { inv } from '../../numerics/linalg/index.ts'
import { trueskillUpdate } from '../../pgm/index.ts'
import { key } from '../../random/index.ts'
import { toArray, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { analyseSpec } from './analysis.ts'
import * as d from './dists/index.ts'
import {
  compileFactorGraph,
  defineModel,
  factorGraphPosterior,
  factorGraphVerdict,
  graphOnSpec,
  ops,
  siteOfId,
  toFactorGraph,
  toStructure,
} from './index.ts'
import type { NamedValue } from './operators.ts'

const run = (g: CompiledGraph): { readonly state: MpState; readonly status: string | undefined } => {
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 300 })
  return { state: r.state as MpState, status: r.status }
}
const graphOf = (spec: FactorGraphSpec | undefined): CompiledGraph =>
  compileGraph(spec as FactorGraphSpec, factorOperators())

// ----- the probit regression of the vertical slice -----

const N = 40
const probitModel = probitDefinition

describe('the probit regression compiled to a factor graph', () => {
  const { X, y } = probitData()
  const bound = probitModel.bind({ X, y })
  const c = toFactorGraph(bound)

  it('expands the plates with the ids of ids.ts: a vector variable, an auxiliary and an observation per row', () => {
    expect(c.issues).toEqual([])
    const g = c.graph as FactorGraphSpec
    expect(g.variables.slice(0, 5).map((v) => v.name)).toEqual(['w', 'y#0[0]', 'y[0]', 'y#0[1]', 'y[1]'])
    expect(g.variables[0]?.family).toEqual({ kind: 'gaussian-mv', size: 3 })
    expect(g.variables[2]).toMatchObject({ family: { kind: 'bernoulli' }, observed: y[0] })
    expect(g.factors.slice(0, 4).map((f) => `${f.id} ${f.operator}`)).toEqual([
      'factor:w prior',
      'factor:y#0[0] dot-constant',
      'factor:y[0] probit',
      'factor:y#0[1] dot-constant',
    ])
    expect(g.factors[1]?.params).toEqual({ a: X[0] })
    expect(c.expanded).toEqual({
      plates: [
        { dim: 'row', size: N },
        { dim: 'feature', size: 3 },
      ],
      variables: 1 + 2 * N,
      factors: 1 + 2 * N,
    })
    expect(new Set(g.factors.map((f) => f.id)).size).toBe(g.factors.length)
    expect(c.auxiliary).toHaveLength(N)
    expect(c.sites.map((s) => [s.site, s.layout, s.posterior])).toEqual([
      ['w', 'packed', true],
      ['y', 'observed', false],
    ])
  })

  it('has EP marginals within 0.05 posterior standard deviations of the exact grid, and its evidence', () => {
    const g = graphOf(c.graph)
    const r = run(g)
    expect(r.status).toBe('converged')
    const grid = probitGrid(X, y, 2)
    const post = factorGraphPosterior(c, g, r.state)
    const mean = toFlat(post.mean('w'))
    const sd = toFlat(post.stddev('w'))
    for (let q = 0; q < 3; q++) {
      expect(Math.abs((mean[q] as number) - (grid.mean[q] as number)) / (grid.sd[q] as number)).toBeLessThan(0.05)
      expect((sd[q] as number) / (grid.sd[q] as number)).toBeGreaterThan(0.9)
      expect((sd[q] as number) / (grid.sd[q] as number)).toBeLessThan(1.1)
    }
    expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - grid.logEvidence)).toBeLessThan(0.05)
    expect(post.sites).toEqual(['w'])
    expect(post.form).toBe('variational')
  })

  it('is the same EP as the hand-built graph of the message-passing slice (a vector prior, a dot product and a probit per row)', () => {
    const by = run(graphOf(c.graph))
    const hand: FactorGraphSpec = {
      variables: [
        { name: 'w', family: { kind: 'gaussian-mv', size: 3 } },
        ...X.map((_, i) => ({ name: `s${i}`, family: { kind: 'gaussian' as const } })),
        ...X.map((_, i) => ({ name: `b${i}`, family: { kind: 'bernoulli' as const }, observed: y[i] as number })),
      ],
      factors: [
        gaussianMvPrior('w', [0, 0, 0], [4, 0, 0, 0, 4, 0, 0, 0, 4]),
        ...X.flatMap((row, i) => [
          { operator: 'dot-constant', args: [`s${i}`, 'w'], params: { a: row } },
          { operator: 'probit', args: [`b${i}`, `s${i}`] },
        ]),
      ],
    }
    const hg = graphOf(hand)
    const hr = run(hg)
    const a = marginalsOf(graphOf(c.graph), by.state)['w']?.['mean'] as number[]
    const b = marginalsOf(hg, hr.state)['w']?.['mean'] as number[]
    for (const [q, v] of a.entries()) expect(Math.abs(v - (b[q] as number))).toBeLessThan(1e-9)
  })

  it('agrees with toStructure(spec) on ids: every graph id maps onto a node or factor of the drawing, and every site is covered', () => {
    const g = c.graph as FactorGraphSpec
    const on = graphOnSpec(g)
    const factorView = toStructure(bound, { view: 'factor' })
    const ids = new Set(factorView.nodes.map((n) => n.id))
    for (const [v, site] of Object.entries(on.variables)) expect(ids.has(site), `${v} -> ${site}`).toBe(true)
    for (const [f, site] of Object.entries(on.factors)) expect(ids.has(site), `${f} -> ${site}`).toBe(true)
    // Every site of the spec (latent and observed) has a variable and a factor in the graph.
    const sites = new Set(Object.values(on.variables))
    for (const n of toStructure(bound).nodes.filter((x) => !x.id.startsWith('data:')))
      expect(sites.has(n.id), n.id).toBe(true)
    expect(new Set(Object.values(on.factors))).toEqual(new Set(['factor:w', 'factor:y']))
    // The plate drawing and the factor drawing name the same variables.
    expect(
      new Set(
        toStructure(bound)
          .nodes.map((n) => n.id)
          .filter((id) => !id.startsWith('data:')),
      ),
    ).toEqual(
      new Set(
        factorView.nodes.filter((n) => !n.id.startsWith('factor:') && !n.id.startsWith('data:')).map((n) => n.id),
      ),
    )
    expect(siteOfId('y#0[17]')).toBe('y')
  })

  it('carries the graph through JSON unchanged, and compiles to the same factors from the rebuilt bound model', () => {
    expect(JSON.parse(JSON.stringify(c.graph))).toEqual(c.graph)
    expect(compileFactorGraph(c).factors).toHaveLength(1 + 2 * N)
  })

  it('reads the state of a run midway as a posterior too (no density yet reads as an unbounded spread)', () => {
    const g = graphOf(c.graph)
    const alg = epAlgorithm(g)
    const s0 = alg.init(key(0))
    const early = factorGraphPosterior(c, g, s0, { rule: 'ep' })
    expect(early.diagnostics['sweeps']).toBe(0)
    expect(toFlat(early.stddev('w')).every((v) => Number.isFinite(v))).toBe(true)
  })
})

// ----- Gaussian models: EP is exact -----

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
  it('uses the regression-row operator, a vector variable and no auxiliary', () => {
    expect(c.issues).toEqual([])
    expect(new Set(c.graph?.factors.map((f) => f.operator))).toEqual(new Set(['prior', 'linear-regression-row']))
    expect(c.auxiliary).toEqual([])
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

describe('affine values that mix a vector latent and a scalar latent, and the logistic link', () => {
  const rows = 10
  const x = Array.from({ length: rows }, (_, i) => [Math.sin(i + 1), Math.cos(2 * i)])
  const yv = x.map((r, i) => 0.7 * (r[0] as number) - 0.4 * (r[1] as number) + 1.5 + 0.2 * Math.sin(5 * i))
  const model = defineModel('intercept', (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f] })
    const w = m.latent('w', d.normal(0, 2), { dims: [f] })
    const b = m.latent('b', d.normal(0, 2))
    m.observe('y', d.normal(ops.add(X.matmul(w), b), 0.5), m.data('y', { dims: [n] }))
  })
  const c = toFactorGraph(model.bind({ X: x, y: yv }))

  it('builds a dot product per row and a sum with the scalar: auxiliaries named by site, element and order', () => {
    expect(c.issues).toEqual([])
    expect(c.auxiliary.slice(0, 3)).toEqual(['y#0[0]', 'y#1[0]', 'y#0[1]'])
    expect(c.graph?.factors.slice(1, 4).map((f) => `${f.id} ${f.operator}`)).toEqual([
      'factor:b prior',
      'factor:y#0[0] dot-constant',
      'factor:y#1[0] plus',
    ])
  })

  it('has the exact posterior means of the linear regression with an intercept (a loopy Gaussian graph keeps the means)', () => {
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-13, maxSweeps: 2000 }), { maxSteps: 2000 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState)
    const design = x.map((row) => [...row, 1])
    const prior = [1 / 4, 1 / 4, 1 / 4]
    const precision = [0, 1, 2].map((i) =>
      [0, 1, 2].map(
        (j) =>
          design.reduce((a, row) => a + (row[i] as number) * (row[j] as number), 0) / 0.25 +
          (i === j ? (prior[i] as number) : 0),
      ),
    )
    const cov = toArray(inv(precision)) as number[][]
    const xty = [0, 1, 2].map((j) => design.reduce((a, row, i) => a + (row[j] as number) * (yv[i] as number), 0) / 0.25)
    const mean = cov.map((row) => row.reduce((a, v, j) => a + v * (xty[j] as number), 0))
    const w = toFlat(post.mean('w'))
    expect(Math.abs((w[0] as number) - (mean[0] as number))).toBeLessThan(1e-6)
    expect(Math.abs((w[1] as number) - (mean[1] as number))).toBeLessThan(1e-6)
    expect(Math.abs(Number(toFlat(post.mean('b'))[0]) - (mean[2] as number))).toBeLessThan(1e-6)
  })

  it('maps probs = sigmoid(eta), logits = eta and the logistic link to the same logistic factors', () => {
    const labels = x.map((r, i) => ((r[0] as number) + 0.3 * Math.sin(i) > 0 ? 1 : 0))
    const make = (likelihood: (eta: ReturnType<typeof ops.add>) => ReturnType<typeof d.bernoulli>) =>
      defineModel('logit', (m) => {
        const n = m.range('row')
        const f = m.range('feature')
        const w = m.latent('w', d.normal(0, 2), { dims: [f] })
        m.observe(
          'y',
          likelihood(m.data('X', { dims: [n, f] }).matmul(w) as never),
          m.data('y', { dims: [n], support: 'binary' }),
        )
      }).bind({ X: x, y: labels })
    const a = toFactorGraph(make((eta) => d.bernoulli({ logits: eta })))
    const b = toFactorGraph(make((eta) => d.bernoulli({ probs: ops.sigmoid(eta) })))
    expect(a.issues).toEqual([])
    expect(new Set(a.graph?.factors.map((f) => f.operator))).toEqual(new Set(['prior', 'dot-constant', 'logistic']))
    expect(b.graph).toEqual(a.graph)
    const g = graphOf(a.graph)
    const r = run(g)
    expect(r.status).toBe('converged')
    expect(Number(toFlat(factorGraphPosterior(a, g, r.state).mean('w'))[0])).toBeGreaterThan(0)
  })
})

describe('a hierarchical Gaussian model with known scales: EP is exact on its tree', () => {
  const y = [28, 8, -3, 7, -1]
  const s = [15, 10, 16, 11, 9]
  const tau = 6
  const model = defineModel('schools', (m) => {
    const j = m.range('school')
    const Y = m.data('y', { dims: [j], role: 'response' })
    const S = m.data('sigma', { dims: [j], support: 'positive' })
    const mu = m.latent('mu', d.normal(0, 10))
    const theta = m.latent('theta', d.normal(mu, tau), { dims: [j] })
    m.observe('y', d.normal(theta, S), Y)
  })
  const c = toFactorGraph(model.bind({ y, sigma: s }))
  it('names the elements theta[j] and links each to mu', () => {
    expect(c.graph?.variables.map((v) => v.name).slice(0, 4)).toEqual(['mu', 'theta[0]', 'theta[1]', 'theta[2]'])
    expect(c.sites.find((x) => x.site === 'theta')).toMatchObject({
      layout: 'elements',
      shape: [5],
      family: 'gaussian',
    })
    expect(c.graph?.factors.find((f) => f.id === 'factor:theta[2]')?.operator).toBe('gaussian')
  })
  it('matches the joint Gaussian conditioned on the observations', () => {
    const g = graphOf(c.graph)
    const r = run(g)
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state)
    // The joint of (mu, theta_1..J) is Gaussian: precision = prior part + likelihood part.
    const J = y.length
    const P: number[][] = Array.from({ length: J + 1 }, () => new Array<number>(J + 1).fill(0))
    const set = (a: number, b: number, v: number): void => {
      ;(P[a] as number[])[b] = v
    }
    const bump = (a: number, b: number, v: number): void => set(a, b, ((P[a] as number[])[b] as number) + v)
    bump(0, 0, 1 / 100)
    for (let i = 0; i < J; i++) {
      bump(0, 0, 1 / tau ** 2)
      set(i + 1, i + 1, 1 / tau ** 2 + 1 / (s[i] as number) ** 2)
      set(0, i + 1, -1 / tau ** 2)
      set(i + 1, 0, -1 / tau ** 2)
    }
    const h = [0, ...y.map((v, i) => v / (s[i] as number) ** 2)]
    const cov = toArray(inv(P)) as number[][]
    const mean = cov.map((row) => row.reduce((a, v, q) => a + v * (h[q] as number), 0))
    expect(Math.abs(Number(toFlat(post.mean('mu'))[0]) - (mean[0] as number))).toBeLessThan(1e-9)
    const th = toFlat(post.mean('theta'))
    const sd = toFlat(post.stddev('theta'))
    for (let i = 0; i < J; i++) {
      expect(Math.abs((th[i] as number) - (mean[i + 1] as number))).toBeLessThan(1e-9)
      expect(Math.abs((sd[i] as number) - Math.sqrt((cov[i + 1] as number[])[i + 1] as number))).toBeLessThan(1e-9)
    }
    expect(post.mean('theta').shape).toEqual([J])
  })
  it('runs variational message passing too: the means are exact and the spread is the mean-field one', () => {
    const g = graphOf(c.graph)
    const r = runAlgorithm(vmpAlgorithm(g, { tol: 1e-12 }), { maxSteps: 500 })
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state as MpState, { rule: 'vmp' })
    const ep = factorGraphPosterior(c, g, run(g).state, { rule: 'ep' })
    expect(Math.abs(Number(toFlat(post.mean('mu'))[0]) - Number(toFlat(ep.mean('mu'))[0]))).toBeLessThan(1e-6)
    expect(Number(toFlat(post.stddev('mu'))[0])).toBeLessThan(Number(toFlat(ep.stddev('mu'))[0]))
  })
})

// ----- TrueSkill -----

describe('a TrueSkill-style model in the builder equals the update of pgm', () => {
  const sigma0 = 25 / 3
  const beta = sigma0 / 2
  const tau = sigma0 / 100
  const players = [
    { mu: 25, sigma: 25 / 3 },
    { mu: 27, sigma: 5 },
  ]
  const model = defineModel('trueskill', (m) => {
    const p = m.range('player')
    const g = m.range('game')
    const mu0 = m.data('mu0', { dims: [p] })
    const sd0 = m.data('sd0', { dims: [p], support: 'positive' })
    const winner = m.data('winner', { dims: [g], support: 'index', indexes: p })
    const loser = m.data('loser', { dims: [g], support: 'index', indexes: p })
    const outcome = m.data('outcome', { dims: [g], support: 'binary' })
    const skill = m.latent('skill', d.normal(mu0, sd0), { dims: [p] })
    const perfW = m.latent('perfW', d.normal(ops.at(skill, winner), beta), { dims: [g] })
    const perfL = m.latent('perfL', d.normal(ops.at(skill, loser), beta), { dims: [g] })
    const diff = m.derived('diff', ops.sub(perfW, perfL))
    void diff
    m.observe('outcome', d.bernoulli({ probs: ops.gt(diff, 0) }), outcome)
  })
  const bound = model.bind({
    mu0: players.map((q) => q.mu),
    sd0: players.map((q) => Math.sqrt(q.sigma ** 2 + tau ** 2)),
    winner: [0],
    loser: [1],
    outcome: [1],
  })
  const c = toFactorGraph(bound)
  it('maps at, a difference and a comparison to affine factors and greater-than', () => {
    expect(c.issues).toEqual([])
    const ops = new Set(c.graph?.factors.map((f) => f.operator))
    expect(ops).toEqual(new Set(['gaussian', 'prior', 'difference', 'greater-than']))
  })
  it('gives the same new skills as trueskillUpdate for one game', () => {
    const g = graphOf(c.graph)
    const r = run(g)
    expect(r.status).toBe('converged')
    const post = factorGraphPosterior(c, g, r.state)
    const [[w], [l]] = trueskillUpdate(
      [players[0] as (typeof players)[number]],
      [players[1] as (typeof players)[number]],
      false,
      { drawProbability: 0 },
    )
    const mean = toFlat(post.mean('skill'))
    const sd = toFlat(post.stddev('skill'))
    expect(Math.abs((mean[0] as number) - (w?.mu as number))).toBeLessThan(1e-8)
    expect(Math.abs((mean[1] as number) - (l?.mu as number))).toBeLessThan(1e-8)
    expect(Math.abs((sd[0] as number) - (w?.sigma as number))).toBeLessThan(1e-8)
    expect(Math.abs((sd[1] as number) - (l?.sigma as number))).toBeLessThan(1e-8)
  })
})

// ----- conjugate and discrete models -----

describe('conjugate observations map to the engine operators and EP is exact', () => {
  const near = (a: number, b: number, tol = 1e-9): void => expect(Math.abs(a - b)).toBeLessThan(tol)

  it('beta-Bernoulli: Beta(a + successes, b + failures), for EP and VMP', () => {
    const coin = defineModel('coin', (m) => {
      m.observe(
        'y',
        d.bernoulli({ probs: m.latent('p', d.beta(2, 3)) }),
        m.data('y', { dims: ['flip'], support: 'binary' }),
      )
    })
    const c = toFactorGraph(coin.bind({ y: [1, 0, 1, 1, 0, 1, 1] }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const ep = factorGraphPosterior(c, g, run(g).state)
    near(Number(toFlat(ep.mean('p'))[0]), (2 + 5) / (2 + 3 + 7))
    near(Number(toFlat(ep.variance('p'))[0]), ((2 + 5) * (3 + 2)) / ((2 + 3 + 7) ** 2 * (2 + 3 + 7 + 1)))
    const vmp = runAlgorithm(vmpAlgorithm(g), { maxSteps: 100 })
    near(Number(toFlat(factorGraphPosterior(c, g, vmp.state as MpState, { rule: 'vmp' }).mean('p'))[0]), 7 / 12)
  })

  it('gamma-Poisson with an exposure, and beta-binomial, one group per element through at', () => {
    const rates = defineModel('rates', (m) => {
      const g = m.range('group')
      const k = m.range('obs')
      const exposure = m.data('exposure', { dims: [k] })
      const code = m.data('code', { dims: [k], support: 'index', indexes: g })
      const lam = m.latent('lam', d.gamma(2, 1), { dims: [g] })
      m.observe('n', d.poisson(ops.mul(exposure, ops.at(lam, code))), m.data('n', { dims: [k], support: 'count' }))
    })
    const c = toFactorGraph(
      rates.bind({ exposure: [1, 2, 1.5, 3], code: [0, 0, 1, 1], n: [3, 5, 2, 7] }, { sizes: { group: 2 } }),
    )
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const post = factorGraphPosterior(c, g, run(g).state)
    // Posterior of a group: Gamma(2 + sum n, 1 + sum exposure).
    const mean = toFlat(post.mean('lam'))
    near(mean[0] as number, (2 + 8) / (1 + 3))
    near(mean[1] as number, (2 + 9) / (1 + 4.5))
    const trials = defineModel('trials', (m) => {
      m.observe('k', d.binomial(10, m.latent('p', d.beta(1, 1))), m.data('k', { dims: ['run'], support: 'count' }))
    })
    const bt = toFactorGraph(trials.bind({ k: [3, 4, 6] }))
    const bg = graphOf(bt.graph)
    near(Number(toFlat(factorGraphPosterior(bt, bg, run(bg).state).mean('p'))[0]), (1 + 13) / (2 + 30))
  })

  it('Dirichlet-categorical: Dirichlet(alpha + counts)', () => {
    const die = defineModel('die', (m) => {
      const pi = m.latent('pi', d.dirichlet([1, 1, 2]))
      m.observe('z', d.categorical({ probs: pi }), m.data('z', { dims: ['roll'], support: 'index' }))
    })
    const c = toFactorGraph(die.bind({ z: [0, 2, 2, 1, 2, 0] }))
    expect(c.issues).toEqual([])
    const g = graphOf(c.graph)
    const mean = toFlat(factorGraphPosterior(c, g, run(g).state).mean('pi'))
    // counts: 2, 1, 3 => alpha = 3, 2, 5, total 10
    for (const [q, v] of [0.3, 0.2, 0.5].entries()) near(mean[q] as number, v)
  })

  it('a Bernoulli network as tables: belief propagation on a tree equals enumeration (explaining away)', () => {
    const net = defineModel('net', (m) => {
      const rain = m.latent('rain', d.bernoulli({ probs: 0.3 }))
      const sprinkler = m.latent('sprinkler', d.bernoulli({ probs: 0.4 }))
      const wetProbs = m.derived(
        'wetProbs',
        ops.where(rain, ops.where(sprinkler, 0.99, 0.8), ops.where(sprinkler, 0.9, 0.05)),
      )
      m.observe('wet', d.bernoulli({ probs: wetProbs }), m.data('wet', { support: 'binary' }))
    })
    const c = toFactorGraph(net.bind({ wet: 1 }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'table'])
    const g = graphOf(c.graph)
    const post = factorGraphPosterior(c, g, run(g).state, { rule: 'bp' })
    // P(rain, sprinkler | wet) by enumeration.
    const joint = (r: number, s: number): number =>
      (r ? 0.3 : 0.7) * (s ? 0.4 : 0.6) * (r ? (s ? 0.99 : 0.8) : s ? 0.9 : 0.05)
    const z = joint(0, 0) + joint(0, 1) + joint(1, 0) + joint(1, 1)
    near(Number(toFlat(post.mean('rain'))[0]), (joint(1, 0) + joint(1, 1)) / z)
    near(Number(toFlat(post.mean('sprinkler'))[0]), (joint(0, 1) + joint(1, 1)) / z)
  })
})

// ----- what is refused -----

describe('what cannot be mapped is refused with the node and the reason', () => {
  const issuesOf = (
    spec: ReturnType<typeof defineModel>,
    data: Record<string, number[] | number>,
  ): ReturnType<typeof toFactorGraph> => toFactorGraph(spec.bind(data as never))

  it('an opaque custom node', () => {
    const m = defineModel('custom', (b) => {
      const mu = b.latent('mu', d.normal(0, 1))
      const warped = b.custom('warp', ([a]) => (a as NamedValue).value, [mu])
      b.observe('y', d.normal(warped, 1), b.data('y', { dims: ['row'] }))
    })
    const c = toFactorGraph(m.bind({ y: [1, 2] }))
    expect(c.graph).toBeUndefined()
    expect(c.opaque).toEqual(['y'])
    expect(c.issues.find((i) => i.node === 'y')).toMatchObject({ kind: 'opaque' })
    expect(c.issues[0]?.reason).toMatch(/custom function 'warp' is opaque/)
    expect(analyseSpec(m).opaque).toEqual(['y'])
  })

  it('a random scale, a non-affine operator, a family without a message family, a constraint, a param and a log-density term', () => {
    const scale = defineModel('scale', (b) => {
      const s = b.latent('s', d.gamma(2, 2))
      b.observe('y', d.normal(0, s), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(scale, { y: [1, 2] }).issues[0]).toMatchObject({ node: 'y', kind: 'operator' })
    expect(issuesOf(scale, { y: [1, 2] }).issues[0]?.reason).toMatch(/scale of 'y' depends on a random value/)

    const nonlinear = defineModel('exp', (b) => {
      const w = b.latent('w', d.normal(0, 1))
      b.observe('y', d.normal(ops.exp(w), 1), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(nonlinear, { y: [1, 2] }).issues[0]?.reason).toMatch(/operator 'exp' applied to a random value/)

    const family = defineModel('hn', (b) => {
      b.observe('y', d.normal(b.latent('s', d.halfNormal(1)), 1), b.data('y', { dims: ['row'] }))
    })
    expect(issuesOf(family, { y: [1] }).issues[0]).toMatchObject({ node: 's', kind: 'family' })

    const constrained = defineModel('cons', (b) => {
      b.observe(
        'y',
        d.normal(b.latent('x', d.normal(0, 1), { constraint: { lower: 0 } }), 1),
        b.data('y', { dims: ['row'] }),
      )
    })
    expect(issuesOf(constrained, { y: [1] }).issues[0]).toMatchObject({ node: 'x', kind: 'constraint' })

    const other = defineModel('other', (b) => {
      b.param('theta', 1)
      const x = b.latent('x', d.normal(0, 1))
      b.factor('penalty', ops.neg(ops.square(x)))
    })
    expect(toFactorGraph(other.bind({})).issues.map((i) => [i.node, i.kind])).toEqual([
      ['theta', 'node'],
      ['penalty', 'node'],
    ])
  })

  it('unbound data, missing values that are not ignored, and a product of two random values', () => {
    const m = defineModel('missing', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }), { missing: 'impute' })
    })
    expect(toFactorGraph(m.bind({ y: [1, Number.NaN, 2] })).issues[0]).toMatchObject({ node: 'y', kind: 'data' })
    const skip = defineModel('skip', (b) => {
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), b.data('y', { dims: ['row'] }), { missing: 'ignore' })
    })
    const c = toFactorGraph(skip.bind({ y: [1, Number.NaN, 2] }))
    expect(c.issues).toEqual([])
    expect(c.sites.find((s) => s.site === 'y')?.variables).toEqual(['y[0]', null, 'y[2]'])
    const square = defineModel('square', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      b.observe('y', d.normal(ops.mul(a, a), 1), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(square.bind({ y: [1] })).issues[0]?.reason).toMatch(/times itself/)
  })

  it('the verdict names the reason, the operators without a rule, and the graph without issues', () => {
    const m = defineModel('p', (b) => {
      const n = b.range('row')
      const X = b.data('X', { dims: [n, 'f'] })
      const w = b.latent('w', d.normal(0, 1), { dims: ['f'] })
      b.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), b.data('y', { dims: [n], support: 'binary' }))
    })
    const c = toFactorGraph(m.bind({ X: [[1, 2]], y: [1] }))
    const g = compileFactorGraph(c)
    expect(factorGraphVerdict(c, g, 'ep').ok).toBe(true)
    // The probit link has a variational rule (the Albert-Chib augmentation), the affine map too.
    expect(factorGraphVerdict(c, g, 'vmp').ok).toBe(true)
    // A hard comparison has none, and says why.
    const hard = defineModel('hard', (b) => {
      const n = b.range('row')
      const X = b.data('X', { dims: [n, 'f'] })
      const w = b.latent('w', d.normal(0, 1), { dims: ['f'] })
      b.observe('y', d.bernoulli({ probs: ops.gt(X.matmul(w), 0) }), b.data('y', { dims: [n], support: 'binary' }))
    })
    const ch = toFactorGraph(hard.bind({ X: [[1, 2]], y: [1] }))
    const gh = compileFactorGraph(ch)
    expect(factorGraphVerdict(ch, gh, 'ep').ok).toBe(true)
    expect(factorGraphVerdict(ch, gh, 'vmp')).toMatchObject({ ok: false, missing: ['vmp message rules'] })
    expect(factorGraphVerdict(ch, gh, 'vmp').reason).toMatch(
      /greater-than has no variational.*KL\(q \|\| p\) is infinite/,
    )
    const bad = toFactorGraph(
      defineModel('bad', (b) => {
        b.observe(
          'y',
          d.normal(b.latent('mu', d.normal(0, 1)), b.latent('s', d.gamma(1, 1))),
          b.data('y', { dims: ['r'] }),
        )
      }).bind({ y: [1] }),
    )
    expect(factorGraphVerdict(bad, undefined, 'ep')).toMatchObject({ ok: false })
    expect(() => compileFactorGraph(bad)).toThrow(/does not compile to a factor graph/)
  })
})

// ----- what a product of two random values, a random precision and a gate map to -----

describe('a product of two random values is the product factor (expectation propagation)', () => {
  const prodModel = defineModel('prod', (b) => {
    const a = b.latent('a', d.normal(1, 1))
    const c = b.latent('c', d.normal(2, 1))
    b.observe('y', d.normal(ops.mul(a, c), 0.5), b.data('y', { dims: ['row'] }))
  })

  it('compiles to a product factor on an auxiliary variable, and EP lands on the grid posterior of the means', () => {
    const c = toFactorGraph(prodModel.bind({ y: [2.5] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'product', 'gaussian'])
    expect(c.auxiliary).toEqual(['y#0[0]'])
    const g = graphOf(c.graph)
    const { state } = run(g)
    const post = factorGraphPosterior(c, g, state)
    // The exact posterior mean of a and c by a two-dimensional grid.
    const logp = (a: number, cc: number): number => -0.5 * (a - 1) ** 2 - 0.5 * (cc - 2) ** 2 - 2 * (2.5 - a * cc) ** 2
    let z = 0
    let ma = 0
    let mc = 0
    for (let i = -80; i <= 80; i++)
      for (let j = -80; j <= 80; j++) {
        const a = i * 0.1
        const cc = j * 0.1 + 2
        const w = Math.exp(logp(a, cc))
        z += w
        ma += w * a
        mc += w * cc
      }
    expect(toFlat(post.mean('a'))[0]).toBeCloseTo(ma / z, 1)
    expect(toFlat(post.mean('c'))[0]).toBeCloseTo(mc / z, 1)
    // The product has no variational rule: the method says so.
    expect(factorGraphVerdict(c, g, 'vmp')).toMatchObject({ ok: false })
  })

  it('a plated product maps per element', () => {
    const plated = defineModel('plated', (b) => {
      const n = b.range('row')
      const a = b.latent('a', d.normal(1, 1), { dims: [n] })
      const c = b.latent('c', d.normal(2, 1), { dims: [n] })
      b.observe('y', d.normal(ops.mul(a, c), 0.5), b.data('y', { dims: [n] }))
    })
    const c = toFactorGraph(plated.bind({ y: [2.5, 1, 3] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'product').length).toBe(3)
    expect(c.auxiliary.length).toBe(3)
  })
})

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

  it('has no expectation propagation rule, and a scale that is not the idiom is refused with it named', () => {
    const c = toFactorGraph(model.bind({ y: ys }))
    expect(factorGraphVerdict(c, graphOf(c.graph), 'ep')).toMatchObject({ ok: false })
    expect(factorGraphVerdict(c, graphOf(c.graph), 'ep').reason).toMatch(/gaussian-gamma-precision/)
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

describe('a categorical gate is the select factor (expectation propagation)', () => {
  const ys = [-3.1, -2.8, -3.3, 3.0, 2.7, 3.2]
  const mixture = defineModel('mixture', (b) => {
    const n = b.range('row')
    const k = b.range('component')
    const mu = b.latent('mu', d.normal(b.data('m0', { dims: [k] }), 4), { dims: [k] })
    const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [n] })
    b.observe('y', d.normal(ops.at(mu, z), 0.7), b.data('y', { dims: [n] }))
  })
  const bound = mixture.bind({ y: ys, m0: [-1, 1] })

  it('compiles to a categorical variable per row and a select factor per row', () => {
    const c = toFactorGraph(bound)
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.filter((f) => f.operator === 'select').length).toBe(ys.length)
    expect(c.sites.find((s) => s.site === 'z')).toMatchObject({ family: 'discrete', layout: 'elements' })
    expect(c.graph?.variables.find((v) => v.name === 'z[0]')?.family).toEqual({ kind: 'discrete', size: 2 })
  })

  it('EP finds the two cluster means and the assignments of a well-separated mixture, as exact enumeration does', () => {
    const c = toFactorGraph(bound)
    const g = graphOf(c.graph)
    const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10, damping: 0.3 }), { maxSteps: 400 })
    const post = factorGraphPosterior(c, g, r.state as MpState)
    // Exact: enumerate the 2^6 assignments; given one, the cluster means are independent Gaussians.
    const prior = [-1, 1]
    const v0 = 16
    const s2 = 0.49
    let total = 0
    const meanSum = [0, 0]
    const zProb = new Array<number>(ys.length).fill(0)
    // EP settles on one of the two labelings (the prior is weak, so the exact posterior has both): condition on the first row in the first component.
    for (let a = 0; a < 2 ** ys.length; a += 2) {
      const members: number[][] = [[], []]
      for (const [i, y] of ys.entries()) (members[(a >> i) & 1] as number[]).push(y)
      let logw = 0
      const m: number[] = []
      for (const kk of [0, 1]) {
        const ysk = members[kk] as number[]
        const p = 1 / v0 + ysk.length / s2
        const h = (prior[kk] as number) / v0 + ysk.reduce((t, v) => t + v, 0) / s2
        m.push(h / p)
        // log marginal likelihood of the members given the Gaussian prior on the mean
        const sy = ysk.reduce((t, v) => t + v, 0)
        const syy = ysk.reduce((t, v) => t + v * v, 0)
        logw +=
          -0.5 * ysk.length * Math.log(2 * Math.PI * s2) -
          0.5 * Math.log(v0 * p) -
          0.5 * (syy / s2 + (prior[kk] as number) ** 2 / v0 - (h * h) / p) +
          0 * sy
      }
      logw += ys.length * Math.log(0.5)
      const w = Math.exp(logw)
      total += w
      meanSum[0] = (meanSum[0] as number) + w * (m[0] as number)
      meanSum[1] = (meanSum[1] as number) + w * (m[1] as number)
      for (let i = 0; i < ys.length; i++) zProb[i] = (zProb[i] as number) + w * ((a >> i) & 1)
    }
    const mu = toFlat(post.mean('mu'))
    expect(Math.abs((mu[0] as number) - (meanSum[0] as number) / total)).toBeLessThan(0.15)
    expect(Math.abs((mu[1] as number) - (meanSum[1] as number) / total)).toBeLessThan(0.15)
    // Rows 0 to 2 sit at the first mean, rows 3 to 5 at the second.
    const pz = marginalsOf(g, r.state as MpState)
    const probsOf = (i: number): number[] => (pz[`z[${i}]`]?.['probs'] ?? []) as number[]
    for (const i of [0, 1, 2]) expect(probsOf(i)[0] as number).toBeGreaterThan(0.9)
    for (const i of [3, 4, 5]) expect(probsOf(i)[1] as number).toBeGreaterThan(0.9)
    for (const i of [1, 4])
      expect(Math.abs((probsOf(i)[1] as number as number) - (zProb[i] as number) / total)).toBeLessThan(0.05)
  })

  it('a Dirichlet latent as the probabilities is the conjugate discrete factor; a Bernoulli gate in where is a select', () => {
    const weights = defineModel('weights', (b) => {
      const n = b.range('row', { size: 3 })
      const pi = b.latent('pi', d.dirichlet([1, 1]))
      b.latent('z', d.categorical({ probs: pi }), { dims: [n] })
    })
    const cw = toFactorGraph(weights.bind({}))
    expect(cw.graph?.factors.filter((f) => f.operator === 'discrete').length).toBe(3)
    const gate = defineModel('gate', (b) => {
      const n = b.range('row')
      const z = b.latent('z', d.bernoulli({ probs: 0.5 }), { dims: [n] })
      const a = b.latent('a', d.normal(3, 1))
      const c = b.latent('c', d.normal(-3, 1))
      b.observe('y', d.normal(ops.where(z, a, c), 0.5), b.data('y', { dims: [n] }))
    })
    const cg = toFactorGraph(gate.bind({ y: [2.8, -3.2, 3.1] }))
    expect(cg.issues).toEqual([])
    expect(cg.graph?.factors.filter((f) => f.operator === 'select').length).toBe(3)
    const gg = graphOf(cg.graph)
    const r = runAlgorithm(epAlgorithm(gg, { tol: 1e-10, damping: 0.3 }), { maxSteps: 400 })
    const m = marginalsOf(gg, r.state as MpState)
    expect((m['z[0]']?.['p'] as number) > 0.9).toBe(true)
    expect((m['z[1]']?.['p'] as number) < 0.1).toBe(true)
  })

  it('a known component, a computed random index and the normal-gamma idioms are refused with their reasons', () => {
    const known = defineModel('known', (b) => {
      const z = b.latent('z', d.categorical({ probs: [0.5, 0.5] }))
      b.observe('y', d.normal(ops.at(b.data('c', { dims: ['k'] }), z), 1), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(known.bind({ c: [0, 1], y: [0.5] })).issues[0]?.reason).toMatch(/known component|at: /)
    const normalGamma = defineModel('ng', (b) => {
      const tau = b.latent('tau', d.gamma(2, 1))
      const mu = b.latent('mu', d.normal(0, ops.div(1, ops.sqrt(ops.mul(tau, 2)))))
      b.observe('y', d.normal(mu, ops.div(1, ops.sqrt(tau))), b.data('y', { dims: ['row'] }))
    })
    const issue = toFactorGraph(normalGamma.bind({ y: [1, 2] })).issues[0]
    expect(issue?.node).toBe('mu')
    expect(issue?.reason).toMatch(/1 \/ sqrt\(tau\)/)
  })
})

describe('the larger and the smaller of two random values are the max and min factors', () => {
  it('compile, run under EP, and refuse a known operand', () => {
    const m = defineModel('hinge', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const c = b.latent('c', d.normal(1, 1))
      b.observe('y', d.normal(ops.maximum(a, c), 0.5), b.data('y', { dims: ['row'] }))
    })
    const c = toFactorGraph(m.bind({ y: [2] }))
    expect(c.issues).toEqual([])
    expect(c.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'prior', 'max', 'gaussian'])
    const g = graphOf(c.graph)
    const { state } = run(g)
    // The exact posterior mean of a by a grid over (a, c), y = max(a, c) observed through N(2, 1/4).
    let z = 0
    let ma = 0
    for (let i = -70; i <= 70; i++)
      for (let j = -70; j <= 70; j++) {
        const a = i * 0.1
        const cc = j * 0.1
        const w = Math.exp(-0.5 * a * a - 0.5 * (cc - 1) ** 2 - 2 * (2 - Math.max(a, cc)) ** 2)
        z += w
        ma += w * a
      }
    expect(toFlat(factorGraphPosterior(c, g, state).mean('a'))[0]).toBeCloseTo(ma / z, 1)
    const low = defineModel('low', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const cc = b.latent('c', d.normal(1, 1))
      b.observe('y', d.normal(ops.minimum(a, cc), 0.5), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(low.bind({ y: [0] })).graph?.factors.some((f) => f.operator === 'min')).toBe(true)
    const clip = defineModel('clip', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      b.observe('y', d.normal(ops.maximum(a, 0), 0.5), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(clip.bind({ y: [1] })).issues[0]?.reason).toMatch(/random value and a known one/)
  })
})
