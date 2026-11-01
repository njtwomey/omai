import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { bocpd, bocpdAlgorithm, bocpdContract, constantHazard } from './changepoint.ts'
import type { ConjugateModel } from './segment-models.ts'

const y = [0.1, -0.2, 0.05, 0.3, 4.1, 3.8, 4.4, 3.9, 4.2, 0.2, -0.1, 0.4]
const mean: ConjugateModel = { kind: 'normal-known-variance', sigma: 0.4, priorMean: 0, priorStd: 3 }

const models: Record<string, { model: ConjugateModel; series: readonly number[] }> = {
  'normal-known-variance': { model: mean, series: y },
  'normal-known-mean': { model: { kind: 'normal-known-mean', mean: 0, alpha: 3, beta: 2 }, series: y },
  'normal-gamma': { model: { kind: 'normal-gamma', mu0: 0, kappa0: 0.5, alpha0: 2, beta0: 1 }, series: y },
  'poisson-gamma': { model: { kind: 'poisson-gamma', shape: 2, rate: 1 }, series: [1, 0, 2, 1, 7, 9, 8, 6, 1, 0, 2] },
  'bernoulli-beta': { model: { kind: 'bernoulli-beta', alpha: 1, beta: 1 }, series: [0, 0, 0, 1, 1, 1, 1, 0, 1, 1] },
  'ar1-known-noise': { model: { kind: 'ar1-known-noise', sigma: 0.5, priorStd: 0.6 }, series: y },
  'trend-known-noise': {
    model: { kind: 'trend-known-noise', sigma: 0.5, priorMean: 0, interceptStd: 2, slopeStd: 0.3 },
    series: y,
  },
}

describe('bocpdAlgorithm', () => {
  it('is bocpd: the result is the algorithm run to the end', () => {
    for (const { model, series } of Object.values(models)) {
      const r = bocpd(series, model, 0.1)
      const s = runAlgorithm(bocpdAlgorithm(series, model, 0.1), { maxSteps: series.length }).state
      expect(s.history.mapRunLength).toEqual(r.mapRunLength)
      expect(s.logEvidence).toBe(r.logEvidence)
      expect(s.t).toBe(series.length)
    }
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    for (const { model, series } of Object.values(models)) {
      const alg = bocpdAlgorithm(series, model, constantHazard(8))
      const whole = runAlgorithm(alg, { maxSteps: series.length }).state
      const half = runAlgorithm(alg, { maxSteps: 5 }).state
      expect(half.t).toBe(5)
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, {
        maxSteps: series.length - 5,
        start: { state: stored, step: 5, serialised: true },
      })
      expect(rest.steps).toBe(series.length)
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    }
  })

  it('logs its contract by path: scalars per step and the run-length row', () => {
    const rec = recorder()
    const alg = bocpdAlgorithm(y, mean, 0.1)
    runAlgorithm(alg, {
      maxSteps: 4,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('bocpd/change-probability')).toHaveLength(4)
    expect(metric('bocpd/map-run-length')).toEqual([1, 2, 3, 4])
    expect(metric('bocpd/hypotheses')).toHaveLength(4)
    const rows = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'bocpd/run-length' ? [e] : []))
    expect(rows.map((e) => e.shape[0])).toEqual([2, 3, 4, 5])
    expect(Object.keys(bocpdContract)).toContain('bocpd/predictive-std')
  })

  it('keeps no rows in the state when asked, and the same numbers', () => {
    const full = bocpd(y, mean, 0.1)
    const lean = bocpd(y, mean, 0.1, { rows: false })
    expect(lean.runLength).toHaveLength(0)
    expect(lean.logEvidence).toBe(full.logEvidence)
    expect(lean.changeProbability).toEqual(full.changeProbability)
  })

  it('rejects a hazard outside (0, 1) and an invalid prior', () => {
    expect(() => bocpdAlgorithm(y, mean, 0)).toThrow()
    expect(() => bocpdAlgorithm(y, { ...mean, sigma: -1 } as ConjugateModel, 0.1)).toThrow()
  })

  it('run lengths of constant data grow by one, whichever the model', () => {
    const flat = new Array<number>(15).fill(2)
    for (const model of [
      mean,
      { kind: 'normal-gamma', mu0: 2, kappa0: 1, alpha0: 2, beta0: 1 } satisfies ConjugateModel,
      { kind: 'poisson-gamma', shape: 2, rate: 1 } satisfies ConjugateModel,
    ]) {
      expect(bocpd(flat, model, 0.01).mapRunLength).toEqual(flat.map((_, t) => t + 1))
    }
  })

  it('the run-length posterior sums to one and the evidence is the product of the predictives', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...Object.keys(models)),
        fc.double({ min: 0.01, max: 0.5, noNaN: true }),
        fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: 3, maxLength: 12 }),
        (name, hazard, noise) => {
          const entry = models[name]
          if (entry === undefined) return false
          // Perturb the model's own series so every kind sees data it can score (counts and bits stay valid).
          const series = entry.series.map((v, i) =>
            name === 'poisson-gamma' || name === 'bernoulli-beta' ? v : v + (noise[i % noise.length] ?? 0),
          )
          const r = bocpd(series, entry.model, hazard)
          const sums = r.runLength.every((row) => Math.abs(row.reduce((a, v) => a + v, 0) - 1) < 1e-9)
          const evidence = r.logPredictive.reduce((a, v) => a + v, 0)
          return sums && Math.abs(evidence - r.logEvidence) < 1e-9 && r.predictiveStd.every((s) => s > 0)
        },
      ),
      { numRuns: 60 },
    )
  })

  it('a threshold prunes hypotheses without changing the evidence much', () => {
    const exact = bocpd(y, mean, 0.1, { threshold: -200 })
    const pruned = bocpd(y, mean, 0.1, { threshold: -8 })
    expect(Math.abs(exact.logEvidence - pruned.logEvidence)).toBeLessThan(0.01)
  })
})
