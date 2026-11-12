// The driver of the compiler on whole models: a bound spec compiled with the ids of `ids.ts`, plates expanded, read
// back as a posterior by site name, agreeing with the drawing of the spec (the probit regression of the vertical
// slice, a hierarchy, affine mixes of a vector and a scalar latent, a TrueSkill game).
import { describe, expect, it } from 'vitest'
import { probitData, probitModel as probitDefinition, probitGrid } from '../../../../testing/probit-slice.ts'
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
} from '../../../message-passing/index.ts'
import { inv } from '../../../numerics/linalg/index.ts'
import { trueskillUpdate } from '../../../pgm/index.ts'
import { key } from '../../../random/index.ts'
import { toArray, toFlat } from '../../../tensor/index.ts'
import { runAlgorithm } from '../../../trace/index.ts'
import * as d from '../dists/index.ts'
import {
  compileFactorGraph,
  defineModel,
  factorGraphPosterior,
  graphOnSpec,
  ops,
  siteOfId,
  toFactorGraph,
  toStructure,
} from '../index.ts'

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
  const c = toFactorGraph(bound, { plates: 'expand' })

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
    const a = toFactorGraph(
      make((eta) => d.bernoulli({ logits: eta })),
      { plates: 'expand' },
    )
    const b = toFactorGraph(
      make((eta) => d.bernoulli({ probs: ops.sigmoid(eta) })),
      { plates: 'expand' },
    )
    expect(a.issues).toEqual([])
    expect(new Set(a.graph?.factors.map((f) => f.operator))).toEqual(new Set(['prior', 'dot-constant', 'logistic']))
    expect(b.graph).toEqual(a.graph)
    // The plated forms are one logistic plate each, equal to one another.
    const pa = toFactorGraph(make((eta) => d.bernoulli({ logits: eta })))
    const pb = toFactorGraph(make((eta) => d.bernoulli({ probs: ops.sigmoid(eta) })))
    expect(pa.graph?.factors.map((f) => f.operator)).toEqual(['prior', 'logistic-plate'])
    expect(pb.graph).toEqual(pa.graph)
    const platedRun = run(graphOf(pa.graph))
    const expandedRun = run(graphOf(a.graph))
    const mp = marginalsOf(graphOf(pa.graph), platedRun.state)['w']?.['mean'] as number[]
    const me = marginalsOf(graphOf(a.graph), expandedRun.state)['w']?.['mean'] as number[]
    for (const [q, v] of mp.entries()) expect(Math.abs(v - (me[q] as number))).toBeLessThan(1e-6)
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
