import { describe, expect, it } from 'vitest'
import { full, sum, toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { defineEstimator } from './estimator.ts'
import { eliminationSchedules } from './feature-selection-options.ts'
import { rfe, rfeAlgorithm, rfecv, rfecvAlgorithm } from './feature-selection-wrappers.ts'
import { rowsOf } from './validation.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const],
  detail: 'full' as const,
}
const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

// Importance is the column sum, so the columns drop in order of their sums.
const bySum = defineEstimator(
  {
    name: 'sums',
    kind: 'regressor',
    fit: (d) => ({ imp: toFlat(sum(d.x as never, 0)) }),
    methods: (p) => ({
      featureImportances: p.imp,
      predict: (x) => full([rowsOf(x)], 0),
    }),
  },
  {},
)
const data = {
  x: [
    [1, 5, 2, 9],
    [1, 5, 2, 9],
    [2, 6, 3, 8],
    [2, 6, 3, 8],
  ],
  y: [0, 0, 0, 0],
}

describe('rfeAlgorithm', () => {
  it('drops one feature per round, fits once more at the end, and logs the elimination', () => {
    const rec = recorder()
    const run = runAlgorithm(rfeAlgorithm(bySum, data, { nFeaturesToSelect: 1 }), { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(4)
    expect(run.status).toBe('completed')
    expect(run.state.support).toEqual([false, false, false, true])
    expect(run.state.ranking).toEqual([4, 2, 3, 1])
    expect(run.state.counts).toEqual([4, 3, 2, 1])
    expect(metricsOf(rec, 'rfe/remaining')).toEqual([3, 2, 1, 1])
    expect(metricsOf(rec, 'rfe/dropped')).toEqual([1, 1, 1, 0])
    const est = rfe({ estimator: bySum, nFeaturesToSelect: 1 }).fit(data)
    expect(est.params.support).toEqual(run.state.support)
    expect(est.params.ranking).toEqual([...run.state.ranking])
  })

  it('resumes from a JSON checkpoint to the same selection', () => {
    const alg = rfeAlgorithm(bySum, data, { nFeaturesToSelect: 1 })
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 2, serialised: true } })
    expect(rest.state.support).toEqual(whole.state.support)
    expect(rest.state.ranking).toEqual(whole.state.ranking)
    expect(rest.state.estimator).toEqual(whole.state.estimator)
    expect(rest.steps).toBe(whole.steps)
  })

  it('takes bites from a schedule table, and scores each round on evaluation data', () => {
    expect(eliminationSchedules.names).toEqual(['fixed', 'fraction'])
    const half = runAlgorithm(
      rfeAlgorithm(bySum, data, { nFeaturesToSelect: 1, schedule: { kind: 'fraction', fraction: 0.5 } }),
      { maxSteps: 10 },
    )
    expect(half.state.counts).toEqual([4, 2, 1])
    const scored = runAlgorithm(
      rfeAlgorithm(bySum, data, { nFeaturesToSelect: 2, evaluation: data, scoring: 'neg_mean_squared_error' }),
      { maxSteps: 10 },
    )
    expect(scored.state.scores).toHaveLength(3)
    expect(scored.state.scores.every((v) => Math.abs(v) < 1e-12)).toBe(true)
  })

  it('is observable as an estimator session: the selection so far at each step', () => {
    const s = rfe({ estimator: bySum, nFeaturesToSelect: 1 }).session(data, { logs: 'all' })
    s.advance({ steps: 1 })
    expect(s.fittedAt(0).params.support).toEqual([true, true, true, true])
    expect(s.fittedAt(1).params.support.filter(Boolean)).toHaveLength(3)
    expect(s.fittedAt(1).params.estimatorSupport).toEqual([true, true, true, true])
  })
})

describe('rfecvAlgorithm', () => {
  const x = Array.from({ length: 8 }, (_, i) => [i, 1, 2 * i, 5])
  const d = { x, y: x.map(() => 0) }
  const options = { cv: 2, scoring: 'neg_mean_squared_error' as const }

  it('scores one fold per step, then eliminates to the chosen count, and resumes', () => {
    const alg = rfecvAlgorithm(bySum, d, options)
    const rec = recorder()
    const whole = runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(whole.steps).toBe(3)
    expect(metricsOf(rec, 'rfecv/fold')).toEqual([0, 1])
    expect(whole.state.final).not.toBeNull()
    expect(metricsOf(rec, 'rfecv/best')).toEqual([1])
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 1, serialised: true } })
    expect(rest.state.final?.support).toEqual(whole.state.final?.support)
    expect(rest.state.foldCounts).toEqual(whole.state.foldCounts)
  })

  it('fits as an estimator with the counts it evaluated', () => {
    const f = rfecv({ estimator: bySum, ...options }).fit(d)
    expect(f.params.nFeatures).toEqual([1, 2, 3, 4])
    expect(f.params.splitScores).toHaveLength(2)
    expect(f.params.support.filter(Boolean)).toHaveLength(1)
  })
})
