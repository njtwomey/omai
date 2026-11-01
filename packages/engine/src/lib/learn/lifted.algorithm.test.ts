import { describe, expect, it } from 'vitest'
import { gaussianNB } from '../../testing/learn.ts'
import { full, item, mean } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { streamUncertainties } from './active-learning-options.ts'
import {
  adaptiveConformalAlgorithm,
  adaptiveConformalInference,
  conformalQuantile,
  crossConformalAlgorithm,
  crossConformalRegressor,
} from './conformal.ts'
import { scoreMemories } from './conformal-options.ts'
import { defineEstimator } from './estimator.ts'
import { streamActiveLearning, streamActiveLearningAlgorithm } from './stream-active-learning.ts'
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

const pool = Array.from({ length: 30 }, (_, i) => [Math.sin(i) * 3, Math.cos(1.7 * i) * 3])
const truth = (r: readonly number[]): number => ((r[0] ?? 0) + (r[1] ?? 0) > 0 ? 1 : 0)
const labeled = {
  x: [
    [-3, -3],
    [3, 3],
    [-2, -1],
    [2, 1],
  ],
  y: [0, 1, 0, 1],
}

describe('streamActiveLearningAlgorithm', () => {
  const options = { labeled, stream: pool, oracle: (t: number) => truth(pool[t] ?? []), threshold: 0.2, budget: 5 }

  it('is what streamActiveLearning runs, one arriving row per step, with its decisions on the tape', () => {
    const rec = recorder()
    const run = runAlgorithm(streamActiveLearningAlgorithm(gaussianNB(), options), {
      maxSteps: 40,
      tracer: rec,
      policy,
    })
    expect(run.steps).toBe(30)
    expect(run.state.queried).toEqual(streamActiveLearning(gaussianNB(), options).queried)
    expect(run.state.queried.length).toBeLessThanOrEqual(5)
    expect(metricsOf(rec, 'stream/row')).toEqual(Array.from({ length: 30 }, (_, i) => i))
    const bought = metricsOf(rec, 'stream/queried')
    expect(bought.reduce((a, b) => a + b, 0)).toBe(run.state.queried.length)
    expect(metricsOf(rec, 'stream/labeled').at(-1)).toBe(run.state.queried.length)
    expect(run.state.x.n).toBe(4 + run.state.queried.length)
  })

  it('resumes from a JSON checkpoint to the same labelled set', () => {
    const alg = streamActiveLearningAlgorithm(gaussianNB(), options)
    const whole = runAlgorithm(alg, { maxSteps: 40 })
    const half = runAlgorithm(alg, { maxSteps: 12 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(streamActiveLearningAlgorithm(gaussianNB(), options), {
      maxSteps: 40,
      start: { state: stored, step: 12, serialised: true },
    })
    expect(rest.state.queried).toEqual(whole.state.queried)
    expect([...rest.state.y]).toEqual([...whole.state.y])
    expect(rest.steps).toBe(whole.steps)
  })

  it('scores the model on evaluation data after each row, and takes the measure from a table', () => {
    const rec = recorder()
    const run = runAlgorithm(
      streamActiveLearningAlgorithm(gaussianNB(), {
        ...options,
        evaluation: { x: pool, y: pool.map(truth) },
        uncertainty: { kind: 'entropy' },
        threshold: 0.5,
      }),
      { maxSteps: 40, tracer: rec, policy },
    )
    expect(run.state.scores).toHaveLength(30)
    expect(metricsOf(rec, 'stream/score')).toHaveLength(30)
    const u = metricsOf(rec, 'stream/uncertainty')
    expect(u.every((v) => v >= 0 && v <= 1 + 1e-12)).toBe(true)
  })

  it('has three measures, each in [0, 1] and larger when less certain', () => {
    expect(streamUncertainties.names).toEqual(['leastConfidence', 'margin', 'entropy'])
    for (const name of streamUncertainties.names) {
      const [sure = 1, unsure = 0] = streamUncertainties.make({ kind: name })([
        [0.95, 0.05],
        [0.5, 0.5],
      ])
      expect(sure).toBeLessThan(unsure)
      expect(unsure).toBeLessThanOrEqual(1 + 1e-12)
    }
  })
})

describe('adaptiveConformalAlgorithm', () => {
  const y = Array.from({ length: 120 }, (_, t) => Math.sin(t) * (1 + t / 40))
  const zero = y.map(() => 0)

  it('is what adaptiveConformalInference runs, with the level and coverage on the tape', () => {
    const options = { alpha: 0.1, gamma: 0.05 }
    const rec = recorder()
    const run = runAlgorithm(adaptiveConformalAlgorithm(zero, y, options), { maxSteps: 200, tracer: rec, policy })
    const r = adaptiveConformalInference(zero, y, options)
    expect(run.steps).toBe(120)
    expect([...run.state.errors]).toEqual([...r.errors])
    expect([...run.state.alphas]).toEqual([...r.alphas])
    expect(metricsOf(rec, 'aci/alpha')).toEqual([...r.alphas])
    expect(metricsOf(rec, 'aci/covered')).toEqual(r.errors.map((e) => 1 - e))
    expect(metricsOf(rec, 'aci/coverage').at(-1)).toBeCloseTo(r.coverage, 12)
    expect(r.coverage).toBeGreaterThan(0.8)
  })

  it('resumes from a JSON checkpoint although early intervals are unbounded', () => {
    const alg = adaptiveConformalAlgorithm(zero, y, { alpha: 0.1, gamma: 0.05 })
    const whole = runAlgorithm(alg, { maxSteps: 200 }).state
    expect(whole.lower.some((v) => !Number.isFinite(v))).toBe(true)
    const half = runAlgorithm(alg, { maxSteps: 7 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 200, start: { state: stored, step: 7, serialised: true } }).state
    expect([...rest.errors]).toEqual([...whole.errors])
    expect([...rest.lower]).toEqual([...whole.lower])
    expect(rest.alphaT).toBeCloseTo(whole.alphaT, 12)
  })

  it('forgets old residuals with a window memory, and the memory is a table', () => {
    expect(scoreMemories.names).toEqual(['all', 'window'])
    expect(scoreMemories.make({ kind: 'window', size: 2 })([1, 2, 3, 4])).toEqual([3, 4])
    const wide = adaptiveConformalInference(zero, y, { alpha: 0.1, gamma: 0.05 })
    const windowed = adaptiveConformalInference(zero, y, {
      alpha: 0.1,
      gamma: 0.05,
      memory: { kind: 'window', size: 10 },
    })
    const viaOld = adaptiveConformalInference(zero, y, { alpha: 0.1, gamma: 0.05, window: 10 })
    expect([...windowed.errors]).toEqual([...viaOld.errors])
    expect([...windowed.upper]).not.toEqual([...wide.upper])
  })

  it('rejects a non-positive gamma and mismatched lengths', () => {
    expect(() => adaptiveConformalAlgorithm(zero, y, { gamma: 0 })).toThrow(/gamma/)
    expect(() => adaptiveConformalAlgorithm([0], [1, 2])).toThrow(/one target/)
  })
})

describe('crossConformalAlgorithm', () => {
  const meanModel = defineEstimator(
    {
      name: 'mean',
      kind: 'regressor',
      fit: (d) => ({ m: item(mean(d.y as never)) }),
      methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }),
    },
    {},
  )
  const data = { x: [[0], [1], [2], [3], [4], [5], [6], [7]], y: [0, 2, 4, 6, 8, 10, 12, 14] }

  it('records the out-of-fold residuals fold by fold, as the regressor uses them', () => {
    const rec = recorder()
    const alg = crossConformalAlgorithm(meanModel, data, { cv: 4 })
    const run = runAlgorithm(alg, { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(4)
    expect(run.state.foldOf).toEqual([0, 0, 1, 1, 2, 2, 3, 3])
    expect(metricsOf(rec, 'cross/fold')).toEqual([0, 1, 2, 3])
    const max = metricsOf(rec, 'cross/maxResidual')
    expect(max.every((v, i) => i === 0 || v >= (max[i - 1] ?? 0))).toBe(true)
    const c = crossConformalRegressor(meanModel, data, { cv: 4, alpha: 0.25 })
    expect(c.quantile).toBe(conformalQuantile(run.state.residuals, 0.25))
  })

  it('resumes from a JSON checkpoint to the same residuals', () => {
    const alg = crossConformalAlgorithm(meanModel, data, { cv: 4 })
    const whole = runAlgorithm(alg, { maxSteps: 10 }).state
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 10, start: { state: stored, step: 2, serialised: true } }).state
    expect([...rest.residuals]).toEqual([...whole.residuals])
    expect(rest.foldOf).toEqual(whole.foldOf)
  })
})
