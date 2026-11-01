import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  boostingBinnings,
  boostingLosses,
  cyclicBoosting,
  explainableBoosting,
  explainableBoostingClassifier,
  explainableBoostingRegressor,
  quantileEdges,
  uniformEdges,
} from './index.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const],
  detail: 'full' as const,
}

// The first feature matters twice as much as the second; the third does nothing.
const x = Array.from({ length: 48 }, (_, i) => [i % 6, Math.floor(i / 6) % 4, (i * 5) % 7])
const y = x.map(([a = 0, b = 0]) => 2 * a + b)

describe('cyclicBoosting', () => {
  const options = { bins: 6, rounds: 20, learningRate: 0.3 }

  it('logs the loss, the round and which feature moved most, the biggest mover being the strongest feature', () => {
    const rec = recorder()
    const booster = cyclicBoosting({ x, y }, options)
    runAlgorithm(booster.algorithm, { maxSteps: 20, tracer: rec, policy, seed: 0 })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    const loss = metric('boost/trainLoss')
    expect(loss).toHaveLength(20)
    expect(loss.every((v, i) => i === 0 || v <= (loss[i - 1] ?? 0) + 1e-12)).toBe(true)
    expect(metric('boost/round').at(-1)).toBe(20)
    expect(metric('boost/feature')[0]).toBe(0)
    const change = rec.events.filter((e) => e.type === 'tensor' && e.name === 'boost/shapeChange')
    expect(change).toHaveLength(20)
  })

  it('resumes from a JSON checkpoint to the same model', () => {
    const { algorithm, model } = cyclicBoosting({ x, y }, { ...options, interactions: 1, pairRounds: 5 })
    const whole = runAlgorithm(algorithm, { maxSteps: 25, seed: 0 }).state
    const half = runAlgorithm(algorithm, { maxSteps: 22, seed: 0 }).state
    const stored = viaJson(algorithm.serialise?.(half) ?? null)
    const rest = runAlgorithm(algorithm, {
      maxSteps: 3,
      seed: 0,
      start: { state: stored, step: 22, serialised: true },
    }).state
    const a = model(whole)
    const b = model(rest)
    expect(b.terms).toHaveLength(a.terms.length)
    for (const [t, term] of a.terms.entries()) {
      const other = b.terms[t]
      expect(other?.features).toEqual(term.features)
      const flatten = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flatten) : [v as number])
      for (const [i, v] of flatten(term.values).entries())
        expect(v).toBeCloseTo(flatten(other?.values)[i] ?? Number.NaN, 12)
    }
  })

  it('takes its binning and loss from option tables', () => {
    expect(boostingBinnings.names).toEqual(['quantile', 'uniform'])
    expect(boostingBinnings.make({ kind: 'uniform' })([0, 1, 2, 3, 4, 8], 4)).toEqual(
      uniformEdges([0, 1, 2, 3, 4, 8], 4),
    )
    expect(boostingBinnings.make(undefined)([1, 2, 3, 4], 2)).toEqual(quantileEdges([1, 2, 3, 4], 2))
    expect(boostingLosses.make({ kind: 'logistic' })).toBe('classification')
    const uniform = explainableBoosting({ x, y }, { ...options, binning: { kind: 'uniform' } })
    const quantile = explainableBoosting({ x, y }, options)
    expect(uniform.task).toBe('regression')
    expect(uniform.terms).toHaveLength(quantile.terms.length)
    const labels = y.map((v) => (v > 6 ? 1 : 0))
    expect(explainableBoosting({ x, y: labels }, { ...options, loss: { kind: 'logistic' } }).task).toBe(
      'classification',
    )
  })
})

describe('explainable boosting estimators', () => {
  it('show the shape functions round by round in a session', () => {
    const s = explainableBoostingRegressor({ bins: 6, rounds: 8, learningRate: 0.3 }).session({ x, y }, { logs: 'all' })
    s.advance({ steps: 8 })
    const early = s.probe(1, 'model/shapeFunctions') as { terms: { values: number[] }[] }
    const late = s.probe(8, 'model/shapeFunctions') as { terms: { values: number[] }[] }
    const range = (v: number[]): number => Math.max(...v) - Math.min(...v)
    expect(range(late.terms[0]?.values ?? [])).toBeGreaterThan(range(early.terms[0]?.values ?? []))
  })

  it('classify and regress', () => {
    const labels = y.map((v) => (v > 6 ? 1 : 0))
    const c = explainableBoostingClassifier({ bins: 6, rounds: 30, learningRate: 0.3 }).fit({ x, y: labels })
    const pred = toFlat(c.predict(x))
    expect(pred.filter((p, i) => p === labels[i]).length).toBeGreaterThan(40)
  })
})
