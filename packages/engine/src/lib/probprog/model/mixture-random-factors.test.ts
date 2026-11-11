// The summed-out mixtures with a random share or random precisions (`normalOutliers`, `normalSpikeSlab`): the switch is a hidden
// Bernoulli variable under a gate of the components, the graph of the explicit-switch form, and EP is close to the exact
// posterior (a brute-force grid over the continuous unknowns, the switches summed out).
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

function run(bound: Parameters<typeof toFactorGraph>[0]) {
  const c = toFactorGraph(bound)
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-8, damping: 0.5 }), { maxSteps: 400 })
  return { c, marginals: marginalsOf(g, r.state as MpState) }
}

const logNormal = (x: number, mean: number, sd: number): number =>
  -0.5 * Math.log(2 * Math.PI * sd * sd) - (0.5 * (x - mean) ** 2) / (sd * sd)
const logSum = (a: number, b: number): number => Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)))
const logGamma = (t: number, shape: number, rate: number): number =>
  shape * Math.log(rate) - lgamma(shape) + (shape - 1) * Math.log(t) - rate * t
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

/** Mean and sd of a coordinate of a grid, from the log weights over all nodes and the coordinate of each node. */
function marginal(logs: readonly number[], at: readonly number[]): { mean: number; sd: number } {
  const top = Math.max(...logs)
  const w = logs.map((l) => Math.exp(l - top))
  const z = w.reduce((a, b) => a + b, 0)
  const mean = w.reduce((a, v, i) => a + (v * (at[i] as number)) / z, 0)
  return { mean, sd: Math.sqrt(w.reduce((a, v, i) => a + (v * ((at[i] as number) - mean) ** 2) / z, 0)) }
}

/** A grid in two coordinates (the second may be a log one, with its Jacobian added by the caller): every node's log weight. */
function grid2(
  xs: readonly number[],
  ys: readonly number[],
  logp: (x: number, y: number) => number,
): { logs: number[]; x: number[]; y: number[] } {
  const logs: number[] = []
  const x: number[] = []
  const y: number[] = []
  for (const a of xs)
    for (const b of ys) {
      logs.push(logp(a, b))
      x.push(a)
      y.push(b)
    }
  return { logs, x, y }
}
const linspace = (lo: number, hi: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

const Y = [1.4, 1.6, 1.2, -3.1, 1.7, 1.5, 5.2, 1.1, 1.8, 1.3, 0.9, 1.6]

/** What a belief says about a positive, unit-interval or real variable: its mean and sd. */
function belief(marginals: Record<string, unknown>, name: string): { mean: number; sd: number } {
  const b = marginals[name] as Record<string, number>
  if (b['variance'] !== undefined) return { mean: b['mean'] as number, sd: Math.sqrt(b['variance']) }
  if (b['shape'] !== undefined)
    return { mean: b['shape'] / (b['rate'] as number), sd: Math.sqrt(b['shape']) / (b['rate'] as number) }
  const [a, c] = [b['a'] as number, b['b'] as number]
  return { mean: a / (a + c), sd: Math.sqrt((a * c) / ((a + c) ** 2 * (a + c + 1))) }
}

function near(
  got: { mean: number; sd: number },
  exact: { mean: number; sd: number },
  meanZ = 0.1,
  ratio: [number, number] = [0.9, 1.1],
): void {
  expect(Math.abs(got.mean - exact.mean) / exact.sd).toBeLessThan(meanZ)
  expect(got.sd / exact.sd).toBeGreaterThan(ratio[0])
  expect(got.sd / exact.sd).toBeLessThan(ratio[1])
}

describe('the outlier mixture likelihood with a random scale or a random share', () => {
  const lik = (v: number, centre: number, sd: number, share: number): number =>
    logSum(Math.log(1 - share) + logNormal(v, centre, sd), Math.log(share) + logNormal(v, centre, 10 * sd))

  it('compiles to one gate of two gaussian-gamma-precision branches over a hidden switch per observation', () => {
    const def = defineModel('outliers-precision', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      const tau = m.latent('tau', d.gamma(2, 1))
      m.observe('y', d.normalOutliers(mu, ops.div(1, ops.sqrt(tau)), 0.2, 10), m.data('y', { dims: [row] }))
    })
    const { c } = run(def.bind({ y: Y }))
    const gates = c.graph?.factors.filter((f) => f.operator === 'gate') ?? []
    expect(gates.length).toBe(Y.length)
    const branches = ((gates[0]?.params ?? {}) as { branches?: { operator: string; params: Record<string, number> }[] })
      .branches
    expect(branches?.map((b) => b.operator)).toEqual(['gaussian-gamma-precision', 'gaussian-gamma-precision'])
    // The outlier branch has the precision tau / width^2.
    expect(branches?.[1]?.params['scale']).toBeCloseTo(0.01, 12)
    // The switches are auxiliary: the posterior reports the model's own variables only.
    expect(c.sites.map((s) => s.site).sort()).toEqual(['mu', 'tau', 'y'])
  })

  it('a random inlier precision with a known share: the centre and the precision are close to the grid', () => {
    const def = defineModel('outliers-precision', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      const tau = m.latent('tau', d.gamma(2, 1))
      m.observe('y', d.normalOutliers(mu, ops.div(1, ops.sqrt(tau)), 0.2, 10), m.data('y', { dims: [row] }))
    })
    const { marginals } = run(def.bind({ y: Y }))
    const g = grid2(linspace(-1, 4, 160), linspace(-3, 3.2, 160), (mu, lt) => {
      const tau = Math.exp(lt)
      return (
        logNormal(mu, 0, 5) + logGamma(tau, 2, 1) + lt + Y.reduce((a, v) => a + lik(v, mu, 1 / Math.sqrt(tau), 0.2), 0)
      )
    })
    near(belief(marginals, 'mu'), marginal(g.logs, g.x))
    near(belief(marginals, 'tau'), marginal(g.logs, g.y.map(Math.exp)))
  })

  it('a random share with a known scale: the centre and the share are close to the grid', () => {
    const def = defineModel('outliers-share', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      const share = m.latent('share', d.beta(2, 5))
      m.observe('y', d.normalOutliers(mu, 0.5, share, 10), m.data('y', { dims: [row] }))
    })
    const { marginals } = run(def.bind({ y: Y }))
    const g = grid2(linspace(-1, 4, 160), linspace(0.002, 0.998, 400), (mu, p) => {
      return logNormal(mu, 0, 5) + logBeta(p, 2, 5) + Y.reduce((a, v) => a + lik(v, mu, 0.5, p), 0)
    })
    near(belief(marginals, 'mu'), marginal(g.logs, g.x))
    near(belief(marginals, 'share'), marginal(g.logs, g.y))
  })

  it('a known scale and a known share is still the exact mixture site (no switch, no gate)', () => {
    const def = defineModel('outliers-known', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      m.observe('y', d.normalOutliers(mu, 0.5, 0.2, 10), m.data('y', { dims: [row] }))
    })
    const c = toFactorGraph(def.bind({ y: Y }))
    expect(c.graph?.factors.some((f) => f.operator === 'gate')).toBe(false)
    expect(c.graph?.factors.some((f) => f.operator === 'mixture-row' || f.operator === 'mixture-site')).toBe(true)
  })

  it('a random width, or a scale that is not the precision idiom, is refused with the reason', () => {
    const width = defineModel('outliers-width', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      const w = m.latent('w', d.gamma(2, 0.2))
      m.observe('y', d.normalOutliers(mu, 0.5, 0.2, w), m.data('y', { dims: [row] }))
    })
    expect(
      toFactorGraph(width.bind({ y: Y }))
        .issues.map((i) => i.reason)
        .join(' '),
    ).toMatch(/width/)
    const scale = defineModel('outliers-scale', (m) => {
      const row = m.range('i')
      const mu = m.latent('mu', d.normal(0, 5))
      const s = m.latent('s', d.gamma(2, 1))
      m.observe('y', d.normalOutliers(mu, s, 0.2, 10), m.data('y', { dims: [row] }))
    })
    expect(
      toFactorGraph(scale.bind({ y: Y }))
        .issues.map((i) => i.reason)
        .join(' '),
    ).toMatch(/precision idiom/)
  })
})

describe('the spike and slab prior with a random inclusion probability or a random slab width', () => {
  const Z = [0.1, 3.2, -0.2, -2.4, 0.15, 4.1]
  const noise = 0.5
  // The coefficient summed out: y_j ~ share N(0, slab^2 + noise^2) + (1 - share) N(0, spike^2 + noise^2).
  const lik = (v: number, share: number, slab: number, spike: number): number =>
    logSum(
      Math.log(share) + logNormal(v, 0, Math.hypot(slab, noise)),
      Math.log(1 - share) + logNormal(v, 0, Math.hypot(spike, noise)),
    )

  it('a random inclusion probability: the share and every coefficient are close to the exact posterior', () => {
    const def = defineModel('spike-share', (m) => {
      const j = m.range('j')
      const share = m.latent('share', d.beta(1.5, 3.5))
      const w = m.latent('w', d.normalSpikeSlab(share, 2, 0.1), { dims: [j] })
      m.observe('y', d.normal(w, noise), m.data('y', { dims: [j] }))
    })
    const { c, marginals } = run(def.bind({ y: Z }))
    expect(c.graph?.factors.filter((f) => f.operator === 'gate').length).toBe(Z.length)
    const ps = linspace(0.001, 0.999, 999)
    const logs = ps.map((p) => logBeta(p, 1.5, 3.5) + Z.reduce((a, v) => a + lik(v, p, 2, 0.1), 0))
    near(belief(marginals, 'share'), marginal(logs, ps), 0.05, [0.9, 1.1])
    // A coefficient: the posterior mean of w_j given the share, averaged over the share (the mixture of the two conditionals).
    const top = Math.max(...logs)
    const wt = logs.map((l) => Math.exp(l - top))
    const norm = wt.reduce((a, b) => a + b, 0)
    const wMean = (j: number): number =>
      ps.reduce((acc, p, k) => {
        const v = Z[j] as number
        const s1 = p * Math.exp(logNormal(v, 0, Math.hypot(2, noise)))
        const s0 = (1 - p) * Math.exp(logNormal(v, 0, Math.hypot(0.1, noise)))
        const r1 = s1 / (s1 + s0)
        const m1 = (v * 4) / (4 + noise * noise)
        const m0 = (v * 0.01) / (0.01 + noise * noise)
        return acc + ((wt[k] as number) / norm) * (r1 * m1 + (1 - r1) * m0)
      }, 0)
    for (const j of [0, 1, 3, 5]) {
      const got = marginals[`w[${j}]`] as { mean: number; variance: number }
      expect(Math.abs(got.mean - wMean(j))).toBeLessThan(0.05)
    }
  })

  it('a random slab precision: the precision is close to the grid', () => {
    const def = defineModel('spike-slab-precision', (m) => {
      const j = m.range('j')
      const tau = m.latent('tau', d.gamma(2, 8))
      const w = m.latent('w', d.normalSpikeSlab(0.4, ops.div(1, ops.sqrt(tau)), 0.1), { dims: [j] })
      m.observe('y', d.normal(w, noise), m.data('y', { dims: [j] }))
    })
    const { c, marginals } = run(def.bind({ y: Z }))
    const branches = (c.graph?.factors.find((f) => f.operator === 'gate')?.params ?? {}) as {
      branches?: { operator: string }[]
    }
    expect(branches.branches?.map((b) => b.operator)).toEqual(['gaussian', 'gaussian-gamma-precision'])
    const lts = linspace(-6, 2, 800)
    const logs = lts.map(
      (lt) =>
        logGamma(Math.exp(lt), 2, 8) + lt + Z.reduce((a, v) => a + lik(v, 0.4, 1 / Math.sqrt(Math.exp(lt)), 0.1), 0),
    )
    near(belief(marginals, 'tau'), marginal(logs, lts.map(Math.exp)), 0.1, [0.85, 1.15])
  })

  it('a known inclusion and known widths keep the exact mixture site; a vector read by a matmul with a random share is refused', () => {
    const known = defineModel('spike-known', (m) => {
      const j = m.range('j')
      const w = m.latent('w', d.normalSpikeSlab(0.3, 2, 0.1), { dims: [j] })
      m.observe('y', d.normal(w, noise), m.data('y', { dims: [j] }))
    })
    const c = toFactorGraph(known.bind({ y: Z }))
    expect(c.graph?.factors.some((f) => f.operator === 'gate')).toBe(false)
    const packed = defineModel('spike-packed', (m) => {
      const row = m.range('row')
      const feature = m.range('feature')
      const share = m.latent('share', d.beta(2, 3))
      const w = m.latent('w', d.normalSpikeSlab(share, 2, 0.1), { dims: [feature] })
      m.observe('y', d.normal(m.data('X', { dims: [row, feature] }).matmul(w), 0.5), m.data('y', { dims: [row] }))
    })
    const X = [
      [1, 0],
      [0, 1],
      [1, 1],
      [0.5, -1],
    ]
    const p = toFactorGraph(packed.bind({ X, y: [1, 2, 3, 0] }))
    expect(p.issues.map((i) => i.reason).join(' ')).toMatch(/joint Gaussian vector/)
  })
})
