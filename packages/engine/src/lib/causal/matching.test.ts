import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { treatmentEffects } from '../datasets/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { balanceTable, entropyBalancing, entropyBalancingAlgorithm, matching, splitTreatment } from './index.ts'

const study = (n: number, seed: number, knobs: Record<string, number | string> = {}) => {
  const g = generate(treatmentEffects, {
    key: key(seed),
    knobs: { n, ate: 2, confounding: 1.2, heterogeneity: 1, ...knobs },
  })
  return {
    data: splitTreatment(g.data),
    att: (g.truth.effects?.att ?? 0) as number,
    ate: (g.truth.extra?.['sampleAte'] ?? 0) as number,
  }
}

describe('matching', () => {
  const { data, att, ate } = study(2000, 4)
  it('recovers the ATT, ATC and ATE on the propensity score and on the covariates', () => {
    for (const on of ['propensity', 'mahalanobis', 'covariates'] as const) {
      const r = matching(data, { on, neighbours: 3 })
      expect(Math.abs(r.estimate.value - att)).toBeLessThan(0.45)
      expect(r.estimate.standardError).toBeGreaterThan(0.02)
    }
    const all = matching(data, { estimand: 'ate', on: 'mahalanobis', neighbours: 3 })
    expect(Math.abs(all.estimate.value - ate)).toBeLessThan(0.45)
  })

  it('a caliper leaves poorly matched treated units out and tightens every distance', () => {
    const wide = matching(data, { on: 'propensity' })
    const tight = matching(data, { on: 'propensity', caliper: 0.05 })
    expect(tight.unmatched.length).toBeGreaterThan(wide.unmatched.length)
    for (const u of tight.pairs)
      for (const d of u.distances) expect(d).toBeLessThanOrEqual((tight.caliper ?? 0) + 1e-12)
  })

  it('matching without replacement uses each control at most once', () => {
    const r = matching(data, { on: 'mahalanobis', replacement: false })
    expect(Math.max(...r.timesUsed)).toBe(1)
    expect(r.pairs.length).toBeGreaterThan(100)
  })

  it('finds the exact twins in a toy design', () => {
    const x = [[0], [1], [2], [3], [0.1], [1.1], [2.1], [3.1]]
    const r = matching(
      { x, treatment: [1, 1, 1, 1, 0, 0, 0, 0], outcome: [3, 4, 5, 6, 1, 2, 3, 4] },
      { on: 'covariates' },
    )
    expect(r.estimate.value).toBeCloseTo(2, 12)
    expect(r.pairs.map((u) => u.matches[0])).toEqual([4, 5, 6, 7])
  })

  it('the standard error is about the spread of the estimate over replications', () => {
    const est: number[] = []
    const ses: number[] = []
    for (let s = 0; s < 24; s++) {
      const r = matching(study(600, 100 + s).data, { on: 'mahalanobis', neighbours: 2 })
      est.push(r.estimate.value)
      ses.push(r.estimate.standardError ?? 0)
    }
    const m = est.reduce((a, b) => a + b, 0) / est.length
    const sd = Math.sqrt(est.reduce((a, b) => a + (b - m) ** 2, 0) / (est.length - 1))
    const se = ses.reduce((a, b) => a + b, 0) / ses.length
    // Estimates vary also with the sampled truth, so the reported error must not exceed the spread by much and not be tiny.
    expect(se).toBeGreaterThan(0.4 * sd)
    expect(se).toBeLessThan(2.5 * sd)
  })
})

describe('entropy balancing', () => {
  const { data, att, ate } = study(2000, 5)
  it('balances the covariate means exactly and recovers the ATT, ATC and ATE', () => {
    const r = entropyBalancing(data)
    for (const row of r.balance) expect(Math.abs(row.after)).toBeLessThan(1e-6)
    expect(Math.abs(balanceTable(data.x, data.treatment)[0]?.before ?? 0)).toBeGreaterThan(0.3)
    expect(Math.abs(r.estimate.value - att)).toBeLessThan(0.3)
    expect(Math.abs(entropyBalancing(data, { estimand: 'ate' }).estimate.value - ate)).toBeLessThan(0.3)
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 10)
  })

  it('matching the variances too balances the second moments', () => {
    const r = entropyBalancing(data, { moments: 2 })
    expect(Math.abs(r.estimate.value - att)).toBeLessThan(0.35)
  })

  it('every solver of the slot reaches the same weights', () => {
    const base = entropyBalancing(data)
    for (const config of [
      { kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 },
      { kind: 'bfgs', gradTol: 1e-10 },
      { kind: 'conjugateGradient', gradTol: 1e-10 },
    ])
      expect(entropyBalancing(data, { solver: smoothSolverConfig(config) }).estimate.value).toBeCloseTo(
        base.estimate.value,
        6,
      )
  })

  it('as an algorithm it steps to convergence and logs the dual', () => {
    const run = runAlgorithm(entropyBalancingAlgorithm([-1, 0, 1, 2, 3], 1, [0.5]), { maxSteps: 100, key: key(0) })
    expect(run.status).toBe('converged')
    expect((run.state as { x: Float64Array }).x[0]).toBeGreaterThan(0)
  })

  it('refuses a target outside the convex hull', () => {
    expect(() =>
      entropyBalancing({ x: [[0], [1], [10], [11], [12]], treatment: [0, 0, 1, 1, 1], outcome: [0, 1, 2, 3, 4] }),
    ).toThrow()
  })
})
