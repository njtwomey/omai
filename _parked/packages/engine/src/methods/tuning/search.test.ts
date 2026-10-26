import { describe, expect, it } from 'vitest'
import { xy } from '../../lib/data/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { defineEstimator } from '../../lib/learn/index.ts'
import { meanSquaredError } from '../../lib/metrics/index.ts'
import { dict, real } from '../../lib/space/index.ts'
import { asTensor, full, type TensorLike } from '../../lib/tensor/index.ts'
import { tuneEstimator } from './search.ts'

/** Predicts the constant a + b: the error is smallest along a + b = 3. */
const shifted = defineEstimator(
  {
    name: 'shifted',
    kind: 'regressor',
    fit: () => ({}),
    methods: (_p, h: { readonly a: number; readonly b: number }) => ({
      predict: (x: TensorLike) => full([asTensor(x).shape[0] ?? 0], h.a + h.b),
    }),
  },
  { a: 0, b: 0 },
)
const data = xy(
  Array.from({ length: 30 }, (_, i) => [i]),
  Array.from({ length: 30 }, (_, i) => 3 + 0.1 * Math.sin(i)),
)
const space = dict({ a: real(-5, 5), b: real(0.01, 10, { log: true }) })

describe('tuneEstimator', () => {
  for (const sampler of ['grid', 'random', 'tpe', 'gp'] as const)
    it(`searches with the ${sampler} sampler and explains every trial`, () => {
      const r = tuneEstimator({ estimator: shifted, data, space, cv: 3, metric: meanSquaredError, sampler, trials: 16 })
      expect(r.higherIsBetter).toBe(false)
      expect(r.study.trials.length).toBe(16)
      expect(r.bestScore).toBeLessThan(2)
      const [trials] = recordsOf(r.exposition, 'searchTrials')
      expect(trials?.trials).toHaveLength(16)
      expect(trials?.parameters).toEqual(['a', 'b'])
      expect(trials?.trials[trials.best ?? -1]?.value).toBe(r.bestScore)
      expect(trials?.trials[0]?.intermediate).toHaveLength(3)
      const [surface] = recordsOf(r.exposition, 'hyperparameterSurface')
      expect(surface?.parameters).toEqual(['a', 'b'])
      expect(surface?.values).toHaveLength(20)
      expect(surface?.logScale).toEqual([false, true])
      expect(surface?.path).toHaveLength(16)
      const [curve] = recordsOf(r.exposition, 'learningCurve')
      expect(curve?.validation.at(-1)).toBe(r.bestScore)
      // The best so far never gets worse.
      const best = curve?.validation ?? []
      for (let i = 1; i < best.length; i++) expect(best[i]).toBeLessThanOrEqual(best[i - 1] ?? Infinity)
    })

  it('draws a cross-validation curve for one parameter, and reports each trial', () => {
    const seen: number[] = []
    const r = tuneEstimator({
      estimator: (p) => shifted.with({ a: (p as { a: number }).a, b: 0 }),
      data,
      space: dict({ a: real(0, 6) }),
      cv: 3,
      metric: meanSquaredError,
      sampler: 'grid',
      trials: 7,
      onTrial: (t) => seen.push(t.exposition().length),
    })
    expect(seen).toHaveLength(7)
    expect(r.bestParams).toEqual({ a: 3 })
    const [cv] = recordsOf(r.exposition, 'crossValidationCurve')
    expect(cv?.values).toEqual([0, 1, 2, 3, 4, 5, 6])
    expect(cv?.best).toBe(3)
    expect(cv?.foldScores?.[0]).toHaveLength(3)
    expect(recordsOf(r.exposition, 'hyperparameterSurface')).toHaveLength(0)
  })

  it('prunes hopeless configurations after the first folds', () => {
    const r = tuneEstimator({
      estimator: shifted,
      data,
      space,
      cv: 5,
      metric: meanSquaredError,
      sampler: 'random',
      trials: 20,
      pruner: 'median',
    })
    const [trials] = recordsOf(r.exposition, 'searchTrials')
    const pruned = trials?.trials.filter((t) => t.state === 'pruned') ?? []
    expect(pruned.length).toBeGreaterThan(0)
    for (const t of pruned) expect(t.intermediate?.length).toBeLessThan(5)
    expect(() => tuneEstimator({ estimator: shifted, data, space, sampler: 'nope' as never })).toThrow(/sampler/)
  })
})
