// The mode weights of EP from several starts against references: a two-cluster mean with unequal priors (the exact posterior by
// enumerating the assignments, which gives the mass of each labelling), the clutter problem (the quadrature of its bimodal
// location) and the benchmark mixture with symmetric priors (the label switching: equal weights, the sorted means of the grid).
// Before a start was set on the port of each label's prior the labels' own messages were divided out of their cavity, every start
// reached the same reading and the mixture had one mode: the second labelling, and the clutter's second mode, were never found.
import { describe, expect, it } from 'vitest'
import { ndtr } from '../../numerics/special/index.ts'
import { key } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { clutterProblem } from '../models/clutter.ts'
import { gaussianMixture } from '../models/mixture.ts'
import { epMultiStartMethod } from './ep-multistart.ts'
import { epSpecMethod } from './mp.ts'

const Phi = (x: number): number => item(ndtr(x))

/** The exact posterior of two cluster means (equal weights, unit noise, a normal prior of each mean): mass of `mu0 < mu1`, moments, evidence. */
function enumerate(
  y: readonly number[],
  priorMean: readonly number[],
  priorSd: number,
): {
  below: number
  mean: number[]
  sd: number[]
  logEvidence: number
} {
  const n = y.length
  const logs: number[] = []
  const parts: { m: number[]; v: number[] }[] = []
  for (let code = 0; code < 2 ** n; code++) {
    let lw = n * Math.log(0.5)
    const m: number[] = []
    const v: number[] = []
    for (const k of [0, 1]) {
      let count = 0
      let total = 0
      let squares = 0
      for (let i = 0; i < n; i++)
        if (((code >> i) & 1) === k) {
          count++
          total += y[i] as number
          squares += (y[i] as number) ** 2
        }
      const p0 = 1 / priorSd ** 2
      const info = p0 + count
      const mean0 = priorMean[k] as number
      const b = p0 * mean0 + total
      const mean = b / info
      lw +=
        -0.5 * count * Math.log(2 * Math.PI) -
        0.5 * Math.log(priorSd ** 2) -
        0.5 * Math.log(info) -
        0.5 * (squares + p0 * mean0 ** 2 - b * mean)
      m.push(mean)
      v.push(1 / info)
    }
    logs.push(lw)
    parts.push({ m, v })
  }
  const top = Math.max(...logs)
  const total = logs.reduce((a, l) => a + Math.exp(l - top), 0)
  let below = 0
  const first = [0, 0]
  const second = [0, 0]
  logs.forEach((l, i) => {
    const w = Math.exp(l - top) / total
    const { m, v } = parts[i] as { m: number[]; v: number[] }
    below += w * Phi(((m[1] as number) - (m[0] as number)) / Math.sqrt((v[0] as number) + (v[1] as number)))
    for (const k of [0, 1]) {
      first[k] = (first[k] as number) + w * (m[k] as number)
      second[k] = (second[k] as number) + w * ((v[k] as number) + (m[k] as number) ** 2)
    }
  })
  return {
    below,
    mean: first,
    sd: first.map((mu, k) => Math.sqrt((second[k] as number) - mu * mu)),
    logEvidence: top + Math.log(total),
  }
}

const Y = [-3.2, -2.5, -3.9, -2.8, -3.1, -2.2, 2.9, 3.4, 2.1, 3.8, 2.6, 3.3]

/** Two cluster means with the prior means given, labels with probability one half. */
function clusters(priorMean: readonly number[], priorSd: number) {
  return defineModel('two-clusters', (m) => {
    const row = m.range('row')
    const k = m.range('component', { size: 2 })
    const y = m.data('y', { dims: [row], role: 'response' })
    const mu = m.latent('mu', d.normal(m.const([...priorMean], ['component']), priorSd), { dims: [k] })
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    m.observe('y', d.normal(mu.at(z), 1), y)
  }).bind({ y: Y }, { sizes: { component: 2 } })
}

describe('the mode weights of the multi-start mixture: a two-cluster mean with unequal priors', () => {
  for (const [name, priorMean, priorSd] of [
    ['a weak asymmetric prior', [-0.5, 0.5], 5],
    ['a prior that prefers one labelling by a few nats', [2, -2], 4],
    ['a strong asymmetric prior', [-3, 3], 3],
  ] as const) {
    it(`finds both labellings and weighs them by the exact mass (${name})`, () => {
      const exact = enumerate(Y, priorMean, priorSd)
      const method = epMultiStartMethod(clusters(priorMean, priorSd))
      const run = session(method, { steps: 40, mode: 'eager', config: { starts: 12, labels: 'permute' } })
      const post = method.posterior?.(run.state, run.config)
      const weights = post?.diagnostics['modeWeights'] as number[]
      expect(weights.length).toBe(2)
      expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
      // The mode with the first mean below the second: its weight is the mass of that half of the space.
      const modes = run.state.runs
        .filter((r) => Number.isFinite(r.logEvidence))
        .reduce<{ evidence: number; first: number; second: number }[]>((all, r) => {
          const mean = r.sites['mu']?.mean ?? []
          if (!all.some((q) => Math.abs(q.evidence - r.logEvidence) < 1e-6))
            all.push({ evidence: r.logEvidence, first: mean[0] ?? 0, second: mean[1] ?? 0 })
          return all
        }, [])
      const mass = modes.filter((m) => m.first < m.second).reduce((a, m) => a + Math.exp(m.evidence), 0)
      const all = modes.reduce((a, m) => a + Math.exp(m.evidence), 0)
      expect(Math.abs(mass / all - exact.below)).toBeLessThan(0.01)
      // The evidence of the model, and the moments of the means as the model writes them (the labelled marginals), are the exact ones.
      expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - exact.logEvidence)).toBeLessThan(0.01)
      const rows = post?.summary().filter((r) => r.site === 'mu') ?? []
      for (const k of [0, 1]) {
        expect(Math.abs((rows[k]?.mean ?? Number.NaN) - (exact.mean[k] as number))).toBeLessThan(
          0.02 * (exact.sd[k] as number),
        )
        expect(Math.abs((rows[k]?.sd ?? Number.NaN) / (exact.sd[k] as number) - 1)).toBeLessThan(0.01)
      }
    })
  }

  it('one run keeps one labelling, with its spread; `sort` reads the components in order whichever labelling a fixed point has', () => {
    const bound = clusters([2, -2], 4)
    const single = epSpecMethod(bound)
    const one = session(single, { steps: 300, mode: 'eager', config: { tol: 1e-9 } })
    const exact = enumerate(Y, [2, -2], 4)
    const alone =
      single
        .posterior?.(one.state, one.config)
        ?.summary()
        .filter((r) => r.site === 'mu') ?? []
    // One run says the first mean is 3 +- 0.4 where the exact marginal is 1.9 +- 2.3: the other labelling has 19% of the mass.
    expect(Math.abs((alone[0]?.mean ?? 0) - (exact.mean[0] as number)) / (exact.sd[0] as number)).toBeGreaterThan(0.4)
    expect((alone[0]?.sd ?? 0) / (exact.sd[0] as number)).toBeLessThan(0.25)
    const method = epMultiStartMethod(bound)
    const run = session(method, { steps: 40, mode: 'eager', config: { starts: 12, labels: 'sort' } })
    const rows =
      method
        .posterior?.(run.state, run.config)
        ?.summary()
        .filter((r) => r.site === 'mu') ?? []
    expect(rows[0]?.mean ?? 0).toBeLessThan(-2.5)
    expect(rows[1]?.mean ?? 0).toBeGreaterThan(2.5)
  })
})

describe('the multi-start mixture on the clutter problem, against its quadrature', () => {
  for (const seed of [0, 2, 3]) {
    it(`finds both modes of the location and the evidence of the model (data ${seed})`, () => {
      const data = seed === 0 ? clutterProblem.data : clutterProblem.generate(key(seed), 12)
      const truth = clutterProblem.truth(data, 'fast')
      const theta = truth.marginals['theta']
      const bound = clutterProblem.bound(data)
      const method = epMultiStartMethod(bound)
      const run = session(method, { steps: 40, mode: 'eager', config: { starts: 24 } })
      const post = method.posterior?.(run.state, run.config)
      const row = post?.summary().find((r) => r.site === 'theta')
      expect(Number(post?.diagnostics['modes'])).toBeGreaterThanOrEqual(2)
      expect(Math.abs((row?.mean ?? Number.NaN) - (theta?.mean ?? 0))).toBeLessThan(0.05 * (theta?.sd ?? 1))
      expect(Math.abs((row?.sd ?? Number.NaN) / (theta?.sd ?? 1) - 1)).toBeLessThan(0.05)
      expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.02)
    })
  }

  it('one EP run reports one mode: the location 7.8 +- 0.7 where the posterior is 6.4 +- 2.3', () => {
    const truth = clutterProblem.truth(undefined, 'fast')
    const single = epSpecMethod(clutterProblem.bound())
    const one = session(single, { steps: 300, mode: 'eager', config: { tol: 1e-9 } })
    const row = single
      .posterior?.(one.state, one.config)
      ?.summary()
      .find((r) => r.site === 'theta')
    expect(Math.abs((row?.mean ?? 0) - (truth.marginals['theta']?.mean ?? 0))).toBeGreaterThan(1)
    expect((row?.sd ?? 1) / (truth.marginals['theta']?.sd ?? 1)).toBeLessThan(0.4)
  })
})

describe('the multi-start mixture on the benchmark mixture with symmetric priors', () => {
  it('weighs the two labellings equally, and `sort` reads the means as the grid does', () => {
    const data = gaussianMixture.generate(key(1), 40)
    const truth = gaussianMixture.truth(data, 'fast')
    const method = epMultiStartMethod(gaussianMixture.bound(data))
    const run = session(method, { steps: 40, mode: 'eager', config: { starts: 16, labels: 'sort' } })
    const post = method.posterior?.(run.state, run.config)
    const weights = post?.diagnostics['modeWeights'] as number[]
    expect(weights.length).toBe(2)
    for (const w of weights) expect(w).toBeCloseTo(0.5, 6)
    const rows = post?.summary().filter((r) => r.site === 'mu') ?? []
    for (const [k, name] of ['mu(1)', 'mu(2)'].entries()) {
      const want = truth.marginals[name]
      expect(Math.abs((rows[k]?.mean ?? Number.NaN) - (want?.mean ?? 0))).toBeLessThan(0.05 * (want?.sd ?? 1))
      expect(Math.abs((rows[k]?.sd ?? Number.NaN) / (want?.sd ?? 1) - 1)).toBeLessThan(0.05)
    }
    // The evidence of one mode is the grid's; the model counts the labelling twice.
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - Math.log(2) - (truth.logEvidence ?? 0))).toBeLessThan(
      0.05,
    )
  })
})

describe('`sort` with a second site that has as many entries as there are components', () => {
  it('sorts by the per-component means and leaves the per-group effects where the model put them', () => {
    // Two clusters and two groups of rows: `b` (one effect per group, declared first) is a vector of the length of the components
    // too. Sorting by the first vector of that length took `b` for the components and moved both vectors the wrong way.
    const groups = Y.map((_, i) => i % 2)
    const spec = defineModel('clusters-and-groups', (m) => {
      const row = m.range('row')
      const k = m.range('component', { size: 2 })
      const group = m.range('group', { size: 2 })
      const y = m.data('y', { dims: [row], role: 'response' })
      const g = m.data('g', { dims: [row], role: 'index', indexes: 'group' })
      const b = m.latent('b', d.normal(m.const([1, -1], ['group']), 0.1), { dims: [group] })
      const mu = m.latent('mu', d.normal(m.const([-0.5, 0.5], ['component']), 4), { dims: [k] })
      const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
      m.observe('y', d.normal(mu.at(z).add(b.at(g)), 1), y)
    }).bind({ y: Y, g: groups }, { sizes: { component: 2, group: 2 } })
    const method = epMultiStartMethod(spec)
    const run = session(method, { steps: 40, mode: 'eager', config: { starts: 12, labels: 'sort' } })
    const rows = method.posterior?.(run.state, run.config)?.summary() ?? []
    const means = (site: string): number[] => rows.filter((r) => r.site === site).map((r) => r.mean)
    // the components in order of their means (about -2.9 and 3.1), the groups as the model has them (about 0.92 and -0.92)
    expect(means('mu')[0]).toBeLessThan(-2)
    expect(means('mu')[1]).toBeGreaterThan(2)
    expect(means('b')[0]).toBeGreaterThan(0.5)
    expect(means('b')[1]).toBeLessThan(-0.5)
  })
})
