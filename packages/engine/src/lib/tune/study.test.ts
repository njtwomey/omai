import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { messages, metricSeries, recorder, runTree, tracePolicy } from '../trace/index.ts'
import {
  bestTrial,
  createStudy,
  exportTrials,
  functionObjective,
  gridSampler,
  medianPruner,
  multiObjective,
  studyAlgorithm,
  studyFromJson,
  studyToJson,
  type Trial,
  tpeSampler,
  tune,
} from './index.ts'

const space = dict({ x: real(-2, 2), k: int(1, 5), act: choice(['a', 'b']) })
const objective = functionObjective((p) => {
  const { x, k, act } = p as { x: number; k: number; act: string }
  return x * x + (k - 3) ** 2 + (act === 'a' ? 0 : 0.5)
})

describe('tune', () => {
  it('runs one trial per step and finds the best', () => {
    const r = tune(objective, space, { nTrials: 30, seed: 2 })
    expect(r.study.trials.length).toBe(30)
    expect(r.steps).toBe(30)
    expect(r.status).toBe('completed')
    expect(r.best).toBe(bestTrial(r.study))
    expect(r.study.trials.map((t) => t.number)).toEqual(Array.from({ length: 30 }, (_, i) => i))
  })
  it('traces trials as child runs with params, reports, decisions and values', () => {
    const rec = recorder()
    const reporting = functionObjective((p, trial) => {
      const x = p as number
      for (let s = 0; s < 4; s++) if (trial.report(s, x + 1 / (s + 1))) return x
      return x
    })
    tune(reporting, real(0, 1), {
      nTrials: 8,
      seed: 1,
      pruner: medianPruner({ nStartupTrials: 2 }),
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    const run = rec.run()
    const tree = runTree(run)
    expect(tree[0]?.children.length).toBe(8)
    expect(messages(run, 'trial').length).toBe(8)
    expect(messages(run, 'params').length).toBe(8)
    expect(messages(run, 'decision').length).toBeGreaterThan(0)
    const pruned = messages(run, 'decision').filter((m) => (m.payload as { prune: boolean }).prune)
    expect(pruned.length).toBeGreaterThan(0)
    expect(metricSeries(run, 'intermediate').values.length).toBeGreaterThan(8)
    expect(metricSeries(run, 'best').values.length).toBe(9)
  })
  it('stops on a NaN objective, a throwing objective and a constant objective', () => {
    const nan = tune(
      functionObjective(() => Number.NaN),
      real(0, 1),
      { nTrials: 100 },
    )
    expect(nan.status).toBe('failed')
    expect(nan.study.trials.length).toBe(20)
    expect(nan.study.trials[0]?.message).toMatch(/NaN/)
    const throwing = tune(
      functionObjective(() => {
        throw new Error('boom')
      }),
      real(0, 1),
      { nTrials: 100, maxFailures: 5 },
    )
    expect(throwing.study.trials.length).toBe(5)
    expect(throwing.study.trials[0]?.message).toBe('boom')
    const constant = tune(
      functionObjective(() => 1),
      real(0, 1),
      { nTrials: 100, sampler: tpeSampler() },
    )
    expect(constant.status).toBe('converged')
    expect(constant.study.trials.length).toBe(20)
    expect(() =>
      tune(
        functionObjective(() => {
          throw new Error('rethrown')
        }),
        real(0, 1),
        { nTrials: 3, catchErrors: false },
      ),
    ).toThrow('rethrown')
  })
  it('records failures for the wrong number of values', () => {
    const r = tune(
      functionObjective(() => [1, 2]),
      real(0, 1),
      { nTrials: 2 },
    )
    expect(r.study.trials.every((t) => t.state === 'failed')).toBe(true)
  })
  it('stops after patience trials without improvement and when the resource budget is spent', () => {
    let calls = 0
    const worsening = functionObjective(() => calls++)
    expect(tune(worsening, real(0, 1), { nTrials: 100, patience: 5 }).study.trials.length).toBe(6)
    const reporting = functionObjective((_p, t) => {
      for (let s = 1; s <= 10; s++) t.report(s, s)
      return 0
    })
    expect(
      tune(reporting, real(0, 1), { nTrials: 100, budget: 35, constantTrials: Infinity }).study.trials.length,
    ).toBe(4)
  })
  it('evaluates enqueued configurations first', () => {
    const r = tune(objective, space, { nTrials: 3, enqueue: [{ x: 0, k: 3, act: 'a' }] })
    expect(r.study.trials[0]?.params).toEqual({ x: 0, k: 3, act: 'a' })
    expect(r.study.trials[0]?.source).toBe('enqueued')
    expect(r.best?.values[0]).toBe(0)
  })
  it('resumes a stored study', () => {
    const first = tune(objective, space, { nTrials: 10, seed: 5, sampler: tpeSampler({ nStartupTrials: 4 }) }).study
    const stored = studyFromJson(JSON.parse(JSON.stringify(studyToJson(first))))
    const resumed = tune(objective, space, {
      nTrials: 5,
      seed: 5,
      sampler: tpeSampler({ nStartupTrials: 4 }),
      study: stored,
    })
    const whole = tune(objective, space, { nTrials: 15, seed: 5, sampler: tpeSampler({ nStartupTrials: 4 }) })
    expect(resumed.study.trials).toEqual(whole.study.trials)
  })
  it('round-trips non-finite values through JSON', () => {
    const t: Trial = { number: 0, params: 0, state: 'complete', values: [Infinity], intermediate: [], source: 'x' }
    const s = createStudy(real(0, 1), { trials: [t] })
    expect(studyFromJson(JSON.parse(JSON.stringify(studyToJson(s)))).trials[0]?.values).toEqual([Infinity])
    expect(() => studyFromJson({ not: 'a study' })).toThrow()
  })
  it('exports trials as rows', () => {
    const r = tune(objective, space, { nTrials: 2, seed: 0 })
    const rows = exportTrials(r.study)
    expect(Object.keys(rows[0] ?? {})).toEqual([
      'number',
      'state',
      'value',
      'params_x',
      'params_k',
      'params_act',
      'source',
    ])
  })
  it('checks spaces, directions and samplers', () => {
    expect(() => tune(objective)).toThrow(/space/)
    expect(() => tune(objective, space, { directions: ['minimize', 'maximize'] })).toThrow(/directions/)
    expect(() => createStudy(space, { directions: [] })).toThrow()
    try {
      tune(multiObjective([objective, objective]), space, {
        sampler: { name: 'one', config: {}, multiObjective: false, ask: () => 0 },
      })
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
    }
  })
  it('serialises its state for checkpoints', () => {
    const alg = studyAlgorithm(objective, space, { sampler: gridSampler({ points: 2 }) })
    const state = alg.init(key(0))
    const back = alg.deserialise?.(alg.serialise?.(state) ?? null)
    expect(back).toEqual(state)
  })
})
