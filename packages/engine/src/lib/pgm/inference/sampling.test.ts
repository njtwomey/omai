import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'

const sprinkler = P.bayesNet([
  { name: 'cloudy', states: ['no', 'yes'], parents: [], cpt: [0.5, 0.5] },
  { name: 'sprinkler', states: ['off', 'on'], parents: ['cloudy'], cpt: [0.5, 0.5, 0.9, 0.1] },
  { name: 'rain', states: ['no', 'yes'], parents: ['cloudy'], cpt: [0.8, 0.2, 0.2, 0.8] },
  { name: 'wet', states: ['no', 'yes'], parents: ['sprinkler', 'rain'], cpt: [1, 0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99] },
])

/** |estimate - exact| within k binomial standard errors (n effective samples). */
function within(estimate: number, exact: number, n: number, k = 4): void {
  expect(Math.abs(estimate - exact)).toBeLessThan(k * Math.sqrt((exact * (1 - exact)) / n) + 1e-9)
}

describe('sampling-based inference', () => {
  const evidence = { wet: 'yes' }
  const exact = P.factorValues(P.variableElimination(sprinkler, ['rain'], { evidence }).posterior)[1] ?? 0

  it('rejection samples are exact posterior draws and the acceptance rate estimates P(evidence)', () => {
    const r = P.rejectionSample(sprinkler, key(1), 6000, evidence)
    within(P.factorValues(P.sampleMarginal(sprinkler, r.samples, ['rain']))[1] ?? 0, exact, r.samples.length)
    within(r.acceptance, Math.exp(P.logPartition(sprinkler, { evidence })), 6000)
  })

  it('likelihood weighting is consistent and reports its effective sample size', () => {
    const r = P.likelihoodWeighting(sprinkler, key(2), 6000, evidence)
    expect(r.effectiveSampleSize).toBeGreaterThan(3000)
    expect(r.effectiveSampleSize).toBeLessThanOrEqual(6000)
    within(
      P.factorValues(P.sampleMarginal(sprinkler, r.samples, ['rain'], r.logWeights))[1] ?? 0,
      exact,
      r.effectiveSampleSize,
    )
    // The mean weight is an unbiased estimate of P(evidence).
    const meanWeight = r.logWeights.reduce((s, w) => s + Math.exp(w), 0) / 6000
    expect(Math.abs(meanWeight - Math.exp(P.logPartition(sprinkler, { evidence })))).toBeLessThan(0.02)
  })

  it('Gibbs sampling matches the exact marginals of a Bayes network and an Ising model', () => {
    const samples = P.gibbsSample(sprinkler, key(3), 4000, { evidence, burnIn: 200 })
    expect(samples.every((s) => s['wet'] === 1)).toBe(true)
    // Correlated draws: allow a wider band.
    within(P.factorValues(P.sampleMarginal(sprinkler, samples, ['rain']))[1] ?? 0, exact, 4000, 8)
    const ising = P.isingModel({ rows: 3, cols: 3, coupling: 0.3, field: { s_1_1: 0.4 } })
    const exactIsing = P.junctionTreeMarginals(ising)
    const draws = P.gibbsSample(ising, key(4), 4000, { burnIn: 200, thin: 2 })
    for (const v of ['s_0_0', 's_1_1', 's_2_2']) {
      within(
        P.factorValues(P.sampleMarginal(ising, draws, [v]))[1] ?? 0,
        P.factorValues(exactIsing[v] as P.Factor)[1] ?? 0,
        4000,
        8,
      )
    }
  })

  it('answers approximate queries by every method', () => {
    for (const method of ['rejection', 'likelihoodWeighting', 'gibbs'] as const) {
      const f = P.approximateQuery(sprinkler, ['rain'], { evidence, method, key: key(5), n: 4000 })
      within(P.factorValues(f)[1] ?? 0, exact, method === 'rejection' ? 2500 : 2000, 8)
    }
    expect(() =>
      P.approximateQuery(P.isingModel({ rows: 1, cols: 2 }), ['s_0_0'], { method: 'rejection', key: key(0) }),
    ).toThrow(/Bayes network/)
    expect(() => P.approximateQuery(sprinkler, ['wet'], { evidence, key: key(0) })).toThrow(/observed/)
  })

  it('traces the log probability of each sweep and is reproducible from its key', () => {
    const rec = recorder()
    const a = P.gibbsSample(sprinkler, key(9), 20, { burnIn: 5, tracer: rec })
    const b = P.gibbsSample(sprinkler, key(9), 20, { burnIn: 5 })
    expect(a).toEqual(b)
    expect(metricSeries(rec.run(), 'logProbability').values.length).toBeGreaterThan(20)
    expect(P.effectiveSampleSize([-Infinity, -Infinity])).toBe(0)
  })
})
