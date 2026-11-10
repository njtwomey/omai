// A bound spec compiled to a factor graph for message passing: the plates are expanded with the ids of `ids.ts`, the
// engine's EP reproduces the exact answers (closed forms where the model is Gaussian or conjugate, an exact grid for the
// probit regression, the TrueSkill update of `pgm`), the same graph is read back as a posterior by site name, the ids
// agree with the drawing of the spec, and what cannot be mapped is refused with the node and the reason.
import { describe, expect, it } from 'vitest'
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
import { logNdtr } from '../../numerics/special/index.ts'
import { trueskillUpdate } from '../../pgm/index.ts'
import { foldIn, key, normal as normalDraws, uniform } from '../../random/index.ts'
import { fromStorage, toArray, toFlat } from '../../tensor/index.ts'
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
function probitData(): { readonly X: number[][]; readonly y: number[] } {
  const k = key(2026)
  const z = toFlat(normalDraws(foldIn(k, 1), [N, 2]))
  const u = toFlat(uniform(foldIn(k, 2), [N]))
  const truth = [0.3, 1.2, -0.8]
  const X = Array.from({ length: N }, (_, i) => [1, z[2 * i] as number, z[2 * i + 1] as number])
  const y = X.map((row, i) => {
    const eta = row.reduce((a, v, j) => a + v * (truth[j] as number), 0)
    return (u[i] as number) < Math.exp(toFlat(logNdtr(eta))[0] as number) ? 1 : 0
  })
  return { X, y }
}

const probitModel = defineModel('probit', (m) => {
  const n = m.range('row')
  const f = m.range('feature')
  const X = m.data('X', { dims: [n, f], role: 'covariate' })
  const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
  const w = m.latent('w', d.normal(0, 2), { dims: [f] })
  m.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
})

/**
 * The posterior mean and standard deviation of three weights under N(0, priorSd^2) priors and a probit likelihood, by a
 * grid: a coarse pass over the prior's range, then two over the region the posterior occupies (exact to the grid's step).
 */
function probitGrid(
  X: number[][],
  y: number[],
  priorSd: number,
): { readonly mean: number[]; readonly sd: number[]; readonly logEvidence: number } {
  const logPosterior = (points: number[][]): Float64Array => {
    const zs = new Float64Array(points.length * X.length)
    points.forEach((w, p) => {
      X.forEach((row, i) => {
        const t = row.reduce((a, v, j) => a + v * (w[j] as number), 0)
        zs[p * X.length + i] = y[i] === 1 ? t : -t
      })
    })
    const logs = toFlat(logNdtr(fromStorage(zs, [zs.length], 'float64')))
    return Float64Array.from(points, (w, p) => {
      let s = -0.5 * (w.reduce((a, v) => a + v * v, 0) / priorSd ** 2) - 1.5 * Math.log(2 * Math.PI * priorSd ** 2)
      for (let i = 0; i < X.length; i++) s += logs[p * X.length + i] as number
      return s
    })
  }
  let lo = [-4 * priorSd, -4 * priorSd, -4 * priorSd]
  let hi = [4 * priorSd, 4 * priorSd, 4 * priorSd]
  let result = { mean: [0, 0, 0], sd: [1, 1, 1], logEvidence: 0 }
  for (const m of [21, 31, 31]) {
    const axes = [0, 1, 2].map((q) =>
      Array.from({ length: m }, (_, i) => (lo[q] as number) + (((hi[q] as number) - (lo[q] as number)) * i) / (m - 1)),
    )
    const cell = [0, 1, 2].reduce((a, q) => a * (((hi[q] as number) - (lo[q] as number)) / (m - 1)), 1)
    const points: number[][] = []
    for (const a of axes[0] as number[])
      for (const b of axes[1] as number[]) for (const c of axes[2] as number[]) points.push([a, b, c])
    const lp = logPosterior(points)
    const top = Math.max(...lp)
    let z = 0
    const m1 = [0, 0, 0]
    const m2 = [0, 0, 0]
    lp.forEach((v, p) => {
      const wt = Math.exp(v - top)
      z += wt
      for (let q = 0; q < 3; q++) {
        const c = (points[p] as number[])[q] as number
        m1[q] = (m1[q] as number) + wt * c
        m2[q] = (m2[q] as number) + wt * c * c
      }
    })
    const mean = m1.map((v) => v / z)
    const sd = m2.map((v, q) => Math.sqrt(v / z - (mean[q] as number) ** 2))
    result = { mean, sd, logEvidence: top + Math.log(z * cell) }
    lo = mean.map((v, q) => v - 7 * (sd[q] as number))
    hi = mean.map((v, q) => v + 7 * (sd[q] as number))
  }
  return result
}

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
    const prod = defineModel('prod', (b) => {
      const a = b.latent('a', d.normal(0, 1))
      const c2 = b.latent('c', d.normal(0, 1))
      b.observe('y', d.normal(ops.mul(a, c2), 1), b.data('y', { dims: ['row'] }))
    })
    expect(toFactorGraph(prod.bind({ y: [1] })).issues[0]?.reason).toMatch(/product of two random values/)
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
    expect(factorGraphVerdict(c, g, 'vmp')).toMatchObject({ ok: false, missing: ['vmp message rules'] })
    expect(factorGraphVerdict(c, g, 'vmp').reason).toMatch(/dot-constant, probit have no variational/)
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
