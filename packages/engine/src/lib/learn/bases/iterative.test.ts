import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { dict, real } from '../../space/index.ts'
import { asTensor, full, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  Log,
  messages,
  metricSeries,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import type { LearnData } from '../estimator.ts'
import { iterative } from './iterative.ts'

interface MeanState {
  readonly i: number
  readonly m: number
}

type MeanVerbs = { readonly predict: (x: TensorLike) => Tensor }

/** The mean of the targets, found by averaging one more row per step: a toy solver with channels and a serialiser. */
function meanAlgorithm(data: LearnData, shrink: number): Algorithm<MeanState> {
  const y = toFlat(asTensor(data.y ?? []))
  return {
    name: 'mean',
    config: { shrink },
    schema: channels({ metrics: { mean: { description: 'the running mean' }, seen: { description: 'rows seen' } } }),
    init: () => ({ i: 0, m: 0 }),
    step: (s, ctx) => {
      const m = s.m + ((y[s.i] ?? 0) - s.m) / (s.i + 1)
      ctx.log('extra/gap', Math.abs(m - s.m))
      return { i: s.i + 1, m: m * (1 - shrink) }
    },
    done: (s) => (s.i >= y.length ? 'converged' : undefined),
    observe: (s) => ({ metrics: { mean: s.m, seen: s.i } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MeanState,
  }
}

const constant = (p: { readonly mean: number }): MeanVerbs => ({
  predict: (x) => full([asTensor(x).shape[0] ?? 0], p.mean),
})

const mean = iterative<{ shrink: number }, MeanState, { mean: number }, MeanVerbs>({
  name: 'running-mean',
  kind: 'regressor',
  defaults: { shrink: 0 },
  hyperSpace: (h) => dict({ shrink: real(0, 0.5, { default: h.shrink }) }),
  algorithm: (data, h) => meanAlgorithm(data, h.shrink),
  params: (s) => ({ mean: s.m }),
  verbs: constant,
  contract: () => ({ 'extra/gap': scalar({ description: 'how far the mean moved' }) }),
  logs: { 'solver/mean': Log() },
})

const data: LearnData = {
  x: [
    [0, 0],
    [1, 1],
    [2, 0],
    [3, 1],
  ],
  y: [1, 2, 6, 3],
}

describe('the iterative base', () => {
  it('fits by running the algorithm as a session, with the solver’s channels as solver/ paths', () => {
    const fitted = mean.fit(data)
    expect(fitted.params).toEqual({ mean: 3 })
    expect(toFlat(fitted.predict([[9, 9]]))).toEqual([3])
    expect(metricSeries(fitted.run, 'solver/mean').values).toEqual([0, 1, 1.5, 3, 3])
    expect(metricSeries(fitted.run, 'solver/seen').values).toEqual([])
    expect(fitted.exposition).toEqual([])
  })

  it('keeps nothing when explain is off, and everything asked for otherwise', () => {
    expect(mean.fit(data, { explain: false }).run.events).toEqual([])
    const fitted = mean.fit(data, { logs: { 'extra/gap': Log(), 'fit/iteration/end': Log() } })
    expect(metricSeries(fitted.run, 'extra/gap').values).toEqual([1, 0.5, 1.5, 0])
    // An event fired during step t is stamped t, like anything logged during that step.
    expect(messages(fitted.run, 'fit/iteration/end').map((m) => m.step)).toEqual([0, 1, 2, 3])
  })

  it('exposes the method and a live session with the kind’s probes', () => {
    const m = mean.method(data)
    expect(Object.keys(m.config.spaces)).toEqual(['shrink'])
    const s = mean.session(data, { key: key(1), logs: 'all' })
    expect(s.t).toBe(0)
    s.advance({ steps: 2 })
    expect(s.probe(2, 'model/params')).toEqual({ mean: 1.5 })
    expect(s.probe(1, 'model/predictions')).toEqual([1, 1, 1, 1])
    const surface = s.probe(2, 'model/decision-surface', { resolution: 4 }) as { readonly kind: string }
    expect(surface.kind).toBe('decisionSurface')
    expect(Object.keys(s.contract)).toContain('solver/mean')
    expect(Object.keys(s.contract)).toContain('extra/gap')
  })

  it('takes hyperparameter changes through with() and runs restarts', () => {
    const shrunk = mean.with({ shrink: 0.5 })
    expect(shrunk.fit(data).params.mean).toBeLessThan(3)
    const restarted = iterative<Record<string, never>, MeanState, { mean: number }, MeanVerbs>({
      name: 'restarted',
      kind: 'regressor',
      defaults: {},
      algorithm: (d, _h, k) => ({ ...meanAlgorithm(d, 0), init: () => ({ i: 0, m: k.words[0] % 7 }) }),
      params: (s) => ({ mean: s.m }),
      verbs: constant,
      restarts: { count: () => 3, better: (a, b) => a.m > b.m },
    })
    expect(restarted.fit(data).params.mean).toBe(3)
  })
})
