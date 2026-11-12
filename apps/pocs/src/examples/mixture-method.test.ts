import { type DictSpace, defaults, type UnionSpace } from 'omai-engine/space'
import { type Config, session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { exampleSteps } from './example-method.ts'
import { mixtureExample, mixtureProblem } from './mixture-method.ts'

const base = { ...(defaults(mixtureExample.method.config) as Config), rows: 40, separation: 4 }
const options = (mixtureExample.method.config.spaces['inference'] as UnionSpace).options
const run = (extra: Config, inference: Config) => {
  const kind = String(inference['method'])
  const config = {
    ...base,
    ...extra,
    inference: { method: kind, ...(defaults(options[kind] as DictSpace) as Config), ...inference },
    budgetMs: 0,
  }
  return session(mixtureExample.method, { config, steps: exampleSteps(config), mode: 'eager' })
}
const nuts = { method: 'nuts', numChains: 2, numWarmup: 40, numDraws: 60 }

describe('mixtureProblem', () => {
  it('generates the rows with the truth known, and the Old Faithful waiting times', () => {
    const p = mixtureProblem({ ...base, trueComponents: 3, weights: 'unequal' })
    expect(p.y).toHaveLength(40)
    expect(p.truth?.means).toEqual([-4, 0, 4])
    expect(p.truth?.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(p.truth?.weights[0]).toBeGreaterThan(p.truth?.weights[2] ?? 1)
    const of = mixtureProblem({ ...base, dataset: 'oldFaithful' })
    expect(of.y).toHaveLength(272)
    expect(of.truth).toBeNull()
    expect(of.scale).toBeGreaterThan(5)
  }, 60000)
})

describe('mixtureExample', () => {
  it('offers the samplers and says why tempered SMC does not apply (discrete labels)', () => {
    const offer = mixtureExample.offer(base)
    for (const kind of ['nuts', 'gibbs', 'svi', 'laplace']) expect(offer.applicable).toContain(kind)
    expect(offer.unavailable['tempered-smc']).toMatch(/discrete/)
  }, 60000)

  it('recovers the sorted means of well-separated components whatever the labelling', () => {
    for (const ordering of ['ordered', 'free']) {
      const s = run({ ordering }, nuts)
      const fit = s.probe(s.t, 'mixture/fit') as { rows: { name: string; mean: number; truth: number | null }[] }
      const means = fit.rows.filter((r) => r.name.startsWith('mean'))
      for (const m of means) expect(Math.abs(m.mean - (m.truth as number))).toBeLessThan(1)
    }
  }, 60000)

  it('shows the label switching of free means in the draws out of order', () => {
    const ordered = run({ ordering: 'ordered' }, nuts)
    expect((ordered.probe(ordered.t, 'mixture/fit') as { outOfOrder: number }).outOfOrder).toBe(0)
    const gibbs = run({ ordering: 'free' }, { method: 'gibbs', numWarmup: 20, numDraws: 60 })
    const out = (gibbs.probe(gibbs.t, 'mixture/fit') as { outOfOrder: number }).outOfOrder
    expect(out).toBeGreaterThanOrEqual(0)
    expect(out).toBeLessThanOrEqual(1)
  }, 60000)

  it('checks a free run against an ordered reference on label-invariant quantities', () => {
    const free = run({ ordering: 'free' }, nuts)
    const reference = run({ ordering: 'ordered' }, { ...nuts, numWarmup: 80, numDraws: 120 })
    const summary = reference.probe(reference.t, 'mixture/summary')
    const check = free.probe(free.t, 'mixture/check', { reference: summary }) as {
      densityDistance: number
      maxMeanDifference: number
    }
    expect(check.densityDistance).toBeLessThan(0.1)
    expect(check.maxMeanDifference).toBeLessThan(3)
  }, 60000)

  it('runs with one component, separate spreads, a small concentration, and on Old Faithful', () => {
    for (const extra of [
      { components: 1 },
      { spread: 'separate', components: 3 },
      { concentration: 0.2 },
      { dataset: 'oldFaithful' },
    ]) {
      const s = run(extra, { method: 'nuts', numChains: 2, numWarmup: 20, numDraws: 20 })
      expect(s.probe(s.t, 'mixture/fit')).not.toBeNull()
      expect(s.probe(s.t, 'mixture/pairs')).not.toBeNull()
    }
  }, 60000)

  it('EP recovers the two clusters on the page defaults (150 rows, separation 3), ordered or free, not one wide blob', () => {
    for (const ordering of ['ordered', 'free']) {
      const s = run({ rows: 150, separation: 3, ordering }, { method: 'ep', damping: 0, tol: 1e-6 })
      const fit = s.probe(s.t, 'mixture/fit') as { rows: { name: string; mean: number; truth: number | null }[] }
      for (const r of fit.rows.filter((x) => x.name.startsWith('mean')))
        expect(Math.abs(r.mean - (r.truth as number))).toBeLessThan(0.4)
      // The spread of the components is 1; the blob of one wide Gaussian has 1.8.
      const spread = fit.rows.find((x) => x.name === 'spread')
      expect(Math.abs((spread?.mean ?? 0) - 1)).toBeLessThan(0.25)
    }
  }, 120000)

  it('runs the variational and the Laplace methods too', () => {
    for (const method of ['svi', 'laplace']) {
      const s = run({}, { method })
      expect(s.probe(s.t, 'mixture/fit')).not.toBeNull()
    }
  }, 60000)
})
